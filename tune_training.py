import argparse
import time

import torch

from config import DEVICE, DTYPE
from model import GPT, get_model_config


DEFAULT_PROFILES = ["tiny_fast", "fast_2060", "modern"]
DEFAULT_BATCHES = [16, 24, 32, 40, 48, 64]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark training throughput for candidate profiles.")
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def benchmark(profile, batch_size, iters, warmup, use_compile):
    config = get_model_config(profile)
    block_size = config["block_size"]

    torch.cuda.empty_cache()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = GPT(config).to(DEVICE).train()
    if use_compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01, fused=(DEVICE == "cuda"))
    x = torch.randint(0, config["vocab_size"], (batch_size, block_size), device=DEVICE)
    y = torch.randint(0, config["vocab_size"], (batch_size, block_size), device=DEVICE)

    try:
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        for _ in range(iters):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        tokens_per_sec = batch_size * block_size * iters / elapsed
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if DEVICE == "cuda" else 0.0
        return {
            "ok": True,
            "profile": profile,
            "batch_size": batch_size,
            "block_size": block_size,
            "params": sum(p.numel() for p in model.parameters()),
            "tokens_per_sec": tokens_per_sec,
            "memory_gb": memory_gb,
        }
    except torch.cuda.OutOfMemoryError:
        return {"ok": False, "profile": profile, "batch_size": batch_size, "error": "OOM"}
    finally:
        del model, optimizer, x, y
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'Profile':<16} {'Batch':>6} {'Block':>6} {'Params':>10} {'tok/s':>12} {'VRAM GB':>8}")
    print("-" * 66)

    results = []
    for profile in args.profiles:
        for batch_size in args.batches:
            result = benchmark(profile, batch_size, args.iters, args.warmup, args.compile)
            results.append(result)
            if result["ok"]:
                print(
                    f"{profile:<16} {batch_size:>6} {result['block_size']:>6} "
                    f"{result['params'] / 1e6:>9.1f}M {result['tokens_per_sec']:>12,.0f} {result['memory_gb']:>8.2f}"
                )
            else:
                print(f"{profile:<16} {batch_size:>6} {'-':>6} {'-':>10} {'OOM':>12} {'-':>8}")

    viable = [r for r in results if r["ok"]]
    if viable:
        best = max(viable, key=lambda r: r["tokens_per_sec"])
        print(
            f"\nBest throughput: {best['profile']} batch={best['batch_size']} "
            f"({best['tokens_per_sec']:,.0f} tok/s, {best['memory_gb']:.2f} GB)"
        )


if __name__ == "__main__":
    main()
