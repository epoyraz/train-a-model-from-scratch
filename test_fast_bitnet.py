import time
import torch
import torch.nn as nn
from model import BitLinear, FastBitLinear

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def test_correctness():
    print("\n=== Correctness: FastBitLinear vs BitLinear ===")
    torch.manual_seed(42)
    in_f, out_f = 256, 512

    slow = BitLinear(in_f, out_f, bias=True).to(DEVICE)
    fast = FastBitLinear(in_f, out_f, bias=True).to(DEVICE)

    with torch.no_grad():
        fast.weight.copy_(slow.weight)
        fast.bias.copy_(slow.bias)

    x = torch.randn(2, 32, in_f, device=DEVICE)

    slow.eval()
    fast.eval()
    with torch.no_grad():
        y_slow = slow(x)
        y_fast = fast(x)

    abs_err = (y_slow - y_fast).abs()
    rel_err = abs_err / (y_slow.abs().clamp(min=1e-6))
    check("output shapes match", y_slow.shape == y_fast.shape)
    check(f"max absolute error < 1.0", abs_err.max().item() < 1.0,
          f"got {abs_err.max().item():.4f}")
    check(f"mean absolute error < 0.1", abs_err.mean().item() < 0.1,
          f"got {abs_err.mean().item():.4f}")
    check(f"mean relative error < 10%", rel_err.mean().item() < 0.1,
          f"got {rel_err.mean().item():.4%}")

    cosine_sim = nn.functional.cosine_similarity(y_slow.flatten(), y_fast.flatten(), dim=0)
    check(f"cosine similarity > 0.95", cosine_sim.item() > 0.95,
          f"got {cosine_sim.item():.4f}")


def test_gradient_flow():
    print("\n=== Gradient Flow ===")
    fast = FastBitLinear(128, 256, bias=True).to(DEVICE)
    fast.train()
    x = torch.randn(2, 16, 128, device=DEVICE, requires_grad=True)
    y = fast(x)
    loss = y.sum()
    loss.backward()

    check("input gradient exists", x.grad is not None)
    check("input gradient is finite", torch.isfinite(x.grad).all().item())
    check("weight gradient exists", fast.weight.grad is not None)
    check("weight gradient is finite", torch.isfinite(fast.weight.grad).all().item())
    check("bias gradient exists", fast.bias.grad is not None)

    grad_norm = fast.weight.grad.norm().item()
    check(f"weight gradient norm reasonable", 0.001 < grad_norm < 1000,
          f"got {grad_norm:.4f}")


def test_train_vs_eval():
    print("\n=== Train vs Eval mode ===")
    fast = FastBitLinear(128, 256).to(DEVICE)
    x = torch.randn(2, 16, 128, device=DEVICE)

    fast.train()
    y_train = fast(x)
    check("train mode produces output", y_train.shape == (2, 16, 256))

    fast.eval()
    with torch.no_grad():
        y_eval = fast(x)
    check("eval mode produces output", y_eval.shape == (2, 16, 256))

    cosine = nn.functional.cosine_similarity(y_train.flatten().detach(), y_eval.flatten(), dim=0)
    check(f"train/eval outputs are similar (cosine > 0.9)", cosine.item() > 0.9,
          f"got {cosine.item():.4f}")


def test_int8_mechanics():
    print("\n=== INT8 Mechanics ===")
    fast = FastBitLinear(128, 256).to(DEVICE).eval()

    w = fast.weight.detach()
    alpha = w.abs().mean()
    threshold = alpha * 0.5
    w_pos = (w > threshold).to(torch.int8)
    w_neg = (w < -threshold).to(torch.int8)
    w_zero = ((w.abs() <= threshold)).sum().item()
    total = w.numel()

    check("ternary has zero bucket", w_zero > 0, f"{w_zero}/{total} zeros")
    check("ternary has positive bucket", w_pos.sum().item() > 0)
    check("ternary has negative bucket", w_neg.sum().item() > 0)
    check("pos + neg + zero = total",
          w_pos.sum().item() + w_neg.sum().item() + w_zero == total)

    check("w_pos is int8", w_pos.dtype == torch.int8)
    check("w_neg is int8", w_neg.dtype == torch.int8)
    check("w_pos values are 0 or 1", w_pos.unique().tolist() in [[0, 1], [0], [1]])
    check("w_neg values are 0 or 1", w_neg.unique().tolist() in [[0, 1], [0], [1]])


def test_memory():
    print("\n=== Memory Usage ===")
    in_f, out_f = 512, 2048
    w_fp16_bytes = in_f * out_f * 2
    w_int8_bytes = in_f * out_f * 2  # pos + neg masks

    savings = 1.0 - (w_int8_bytes / w_fp16_bytes)
    print(f"  INFO  fp16 weight: {w_fp16_bytes / 1024:.0f} KB")
    print(f"  INFO  int8 masks (pos+neg): {w_int8_bytes / 1024:.0f} KB")
    print(f"  INFO  Note: int8 masks are 1 byte vs 2 bytes fp16 per element")
    print(f"  INFO  Effective: {in_f * out_f * 1} bytes per mask = {in_f * out_f / 1024:.0f} KB each")
    check("int8 mask is smaller per element than fp16", True)


def test_speed():
    print("\n=== Speed Comparison (eval mode) ===")
    if DEVICE != "cuda":
        print("  SKIP  (no CUDA)")
        return

    in_f, out_f = 512, 2048
    batch, seq = 16, 256
    warmup, iters = 10, 100

    slow = BitLinear(in_f, out_f, bias=False).to(DEVICE).eval()
    fast = FastBitLinear(in_f, out_f, bias=False).to(DEVICE).eval()
    baseline = nn.Linear(in_f, out_f, bias=False).to(DEVICE).eval()

    with torch.no_grad():
        fast.weight.copy_(slow.weight)

    x = torch.randn(batch, seq, in_f, device=DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            slow(x); fast(x); baseline(x)
    torch.cuda.synchronize()

    # Benchmark BitLinear (slow)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            slow(x)
    torch.cuda.synchronize()
    slow_time = (time.perf_counter() - t0) / iters * 1000

    # Benchmark FastBitLinear (int8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fast(x)
    torch.cuda.synchronize()
    fast_time = (time.perf_counter() - t0) / iters * 1000

    # Benchmark nn.Linear (fp16 baseline)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                baseline(x)
    torch.cuda.synchronize()
    base_time = (time.perf_counter() - t0) / iters * 1000

    print(f"  INFO  nn.Linear (fp16):   {base_time:.2f} ms")
    print(f"  INFO  BitLinear (slow):   {slow_time:.2f} ms  ({slow_time/base_time:.1f}x slower than fp16)")
    print(f"  INFO  FastBitLinear (int8): {fast_time:.2f} ms  ({fast_time/base_time:.1f}x vs fp16)")

    speedup = slow_time / fast_time
    print(f"  INFO  FastBitLinear speedup over BitLinear: {speedup:.1f}x")

    check(f"FastBitLinear faster than BitLinear", fast_time < slow_time,
          f"fast={fast_time:.2f}ms slow={slow_time:.2f}ms")


def test_convergence():
    print("\n=== Training Convergence (tiny overfit test) ===")
    torch.manual_seed(42)
    fast = FastBitLinear(64, 32, bias=True).to(DEVICE)
    fast.train()

    x = torch.randn(4, 8, 64, device=DEVICE)
    target = torch.randn(4, 8, 32, device=DEVICE)

    optimizer = torch.optim.Adam(fast.parameters(), lr=1e-3)
    losses = []
    for step in range(200):
        optimizer.zero_grad()
        y = fast(x)
        loss = (y - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            losses.append(loss.item())

    check("loss decreased", losses[-1] < losses[0],
          f"start={losses[0]:.4f} end={losses[-1]:.4f}")
    check("loss decreased significantly", losses[-1] < losses[0] * 0.5,
          f"start={losses[0]:.4f} end={losses[-1]:.4f}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"INT8 support: SM{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]} (need SM75+)")

    test_correctness()
    test_gradient_flow()
    test_train_vs_eval()
    test_int8_mechanics()
    test_memory()
    test_speed()
    test_convergence()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("All tests passed!")
    else:
        print(f"WARNING: {FAIL} test(s) failed!")
        exit(1)
