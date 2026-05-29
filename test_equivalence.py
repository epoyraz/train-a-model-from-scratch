"""Equivalence guards: prove the *fast* paths match their *reference* paths.

These don't test that a technique's math is "nice" (test_techniques.py does that) —
they test that an optimization didn't change the result:
  1. chunked cross-entropy   == dense cross-entropy        (value + gradients)
  2. KV-cache incremental    == dense full forward         (next-token logits)
  3. speculative (greedy)    == autoregressive (greedy)    (token sequence)
  4. torch.compile           == eager                      (logits)  [best-effort]
"""
import torch

from config import DEVICE
from model import GPT, chunked_cross_entropy

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


SMALL = {
    "vocab_size": 256, "block_size": 32, "n_embd": 64, "n_head": 4, "n_layer": 2,
    "use_rope": True, "n_kv_head": 2, "use_swiglu": True, "use_rmsnorm": True,
}
MTP = {**SMALL, "use_mtp": True, "mtp_heads": 2, "mtp_weight": 0.1}


def test_chunked_loss():
    print("\n=== Chunked CE == dense CE (value + grad) ===")
    for label, cfg in [("no-MTP", SMALL), ("MTP", MTP)]:
        torch.manual_seed(0)
        dense = GPT(cfg).to(DEVICE)
        chunked = GPT({**cfg, "use_chunked_loss": True, "loss_chunk_size": 16}).to(DEVICE)
        chunked.load_state_dict(dense.state_dict())
        x = torch.randint(0, cfg["vocab_size"], (4, cfg["block_size"]), device=DEVICE)
        y = torch.randint(0, cfg["vocab_size"], (4, cfg["block_size"]), device=DEVICE)

        _, l_dense = dense(x, y)
        _, l_chunked = chunked(x, y)
        check(f"[{label}] loss matches", torch.allclose(l_dense, l_chunked, atol=1e-4, rtol=1e-4),
              f"dense {l_dense.item():.6f} vs chunked {l_chunked.item():.6f}")

        l_dense.backward(); l_chunked.backward()
        gd = dense.lm_head.weight.grad
        gc = chunked.lm_head.weight.grad
        check(f"[{label}] lm_head grad matches", torch.allclose(gd, gc, atol=1e-4, rtol=1e-3),
              f"max diff {(gd - gc).abs().max().item():.2e}")


def test_kv_cache_equiv():
    print("\n=== KV-cache incremental == dense forward (logits) ===")
    torch.manual_seed(0)
    cfg = SMALL
    m = GPT(cfg).to(DEVICE).eval()
    seq = torch.randint(0, cfg["vocab_size"], (1, 20), device=DEVICE)
    with torch.no_grad():
        dense_logits = m(seq)[0][:, -1, :]  # predict token after full seq, dense
        # incremental: prefill all but last, then one cached step on the last token
        logits, _, caches, seq_len = m._prefill_generation(seq[:, :-1], use_turboquant=False)
        logits, _, caches, seq_len = m._advance_generation_state(
            seq, seq[:, -1:], caches, seq_len, False
        )
        cached_logits = logits[:, -1, :]
    check("incremental logits match dense", torch.allclose(dense_logits, cached_logits, atol=1e-3, rtol=1e-3),
          f"max diff {(dense_logits - cached_logits).abs().max().item():.2e}")


def test_speculative_equiv():
    print("\n=== Speculative (greedy) == autoregressive (greedy) ===")
    torch.manual_seed(0)
    m = GPT(MTP).to(DEVICE).eval()
    prompt = torch.randint(0, MTP["vocab_size"], (1, 5), device=DEVICE)
    with torch.no_grad():
        ar = m.generate(prompt.clone(), 40, temperature=0.0, speculative=False)
        sp = m.generate(prompt.clone(), 40, temperature=0.0, speculative=True)
    n = min(ar.shape[1], sp.shape[1])
    mism = (ar[0, :n] != sp[0, :n]).sum().item()
    check("same length", ar.shape == sp.shape, f"{ar.shape} vs {sp.shape}")
    check("greedy tokens match autoregressive", mism == 0, f"{mism}/{n} mismatches")


def test_compile_equiv():
    print("\n=== torch.compile == eager (logits) [best-effort] ===")
    if DEVICE != "cuda":
        print("  SKIP  (needs CUDA)")
        return
    try:
        from msvc_env import ensure_msvc_env
        ensure_msvc_env(verbose=False)
        torch.manual_seed(0)
        m = GPT(SMALL).to(DEVICE).eval()
        x = torch.randint(0, SMALL["vocab_size"], (2, SMALL["block_size"]), device=DEVICE)
        with torch.no_grad():
            eager = m(x)[0]
            mc = torch.compile(m)
            comp = mc(x)[0]
        check("compiled logits match eager", torch.allclose(eager, comp, atol=1e-3, rtol=1e-3),
              f"max diff {(eager - comp).abs().max().item():.2e}")
    except Exception as e:
        print(f"  SKIP  (compile unavailable: {type(e).__name__})")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    test_chunked_loss()
    test_kv_cache_equiv()
    test_speculative_equiv()
    test_compile_equiv()
    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)
    print("All equivalence guards passed!")
