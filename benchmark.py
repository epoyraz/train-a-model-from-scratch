import time
import torch
from model import GPT, BASE_CONFIG, MHC_CONFIG, BITNET_CONFIG, MHC_BITNET_CONFIG, ALL_CONFIG
from config import DEVICE, DTYPE
BATCH_SIZE = 16
SEQ_LEN = 256
WARMUP = 5
ITERS = 50
GEN_TOKENS = 64


def benchmark_forward(name, config):
    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, config["vocab_size"], (BATCH_SIZE, SEQ_LEN), device=DEVICE)

    for _ in range(WARMUP):
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
            model(x, x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    total_loss = 0.0
    for _ in range(ITERS):
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
            _, loss = model(x, x)
        total_loss += loss.item()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = (elapsed / ITERS) * 1000
    tokens_per_sec = BATCH_SIZE * SEQ_LEN * ITERS / elapsed
    avg_loss = total_loss / ITERS
    mem_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None

    return {
        "name": name,
        "params": n_params,
        "ms_per_step": ms_per_step,
        "tokens_per_sec": tokens_per_sec,
        "loss_at_init": avg_loss,
        "mem_mb": mem_mb,
    }


def benchmark_generate(name, config):
    config = {**config, "use_turboquant": False}
    config_tq = {**config, "use_turboquant": True, "turboquant_bits": 4}

    results = {}
    for label, cfg in [(name, config), (f"{name}+tq", config_tq)]:
        model = GPT(cfg).to(DEVICE).eval()
        prompt = torch.randint(0, cfg["vocab_size"], (1, 32), device=DEVICE)

        for _ in range(3):
            with torch.no_grad():
                model.generate(prompt, max_new_tokens=16)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                model.generate(prompt, max_new_tokens=GEN_TOKENS)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        tokens_per_sec = 10 * GEN_TOKENS / elapsed
        mem_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0
        torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
        results[label] = {"tokens_per_sec": tokens_per_sec, "mem_mb": mem_mb}

    return results


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Benchmark: {BATCH_SIZE}x{SEQ_LEN} tokens, {ITERS} iters\n")

    configs = [
        ("base", BASE_CONFIG),
        ("mhc", MHC_CONFIG),
        ("bitnet", BITNET_CONFIG),
        ("mhc+bitnet", MHC_BITNET_CONFIG),
    ]

    # --- Training benchmark ---
    print("=" * 80)
    print("TRAINING (forward pass)")
    print("=" * 80)
    header = f"{'Config':<14} {'Params':>12} {'ms/step':>10} {'tok/s':>12} {'Init Loss':>10} {'VRAM MB':>10}"
    print(header)
    print("-" * 80)

    results = []
    for name, cfg in configs:
        r = benchmark_forward(name, cfg)
        results.append(r)
        print(f"{r['name']:<14} {r['params']:>12,} {r['ms_per_step']:>10.1f} {r['tokens_per_sec']:>12,.0f} {r['loss_at_init']:>10.2f} {r['mem_mb']:>10.0f}")

    base = results[0]
    print(f"\n{'Config':<14} {'Speed vs base':>14} {'Memory vs base':>15}")
    print("-" * 45)
    for r in results:
        speed = base["ms_per_step"] / r["ms_per_step"]
        mem = r["mem_mb"] / base["mem_mb"] if base["mem_mb"] > 0 else 0
        print(f"{r['name']:<14} {speed:>13.2f}x {mem:>14.2f}x")

    # --- Generation benchmark ---
    print(f"\n{'=' * 80}")
    print("GENERATION (inference)")
    print("=" * 80)
    print(f"{'Config':<14} {'tok/s':>10} {'VRAM MB':>10}")
    print("-" * 40)

    for name, cfg in configs:
        gen = benchmark_generate(name, cfg)
        for label, stats in gen.items():
            print(f"{label:<14} {stats['tokens_per_sec']:>10.1f} {stats['mem_mb']:>10.0f}")


if __name__ == "__main__":
    main()
