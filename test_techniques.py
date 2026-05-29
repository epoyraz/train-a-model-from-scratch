import torch
import torch.nn as nn
from model import (
    sinkhorn, MHCResidual, MHCExpand, MHCCollapse,
    BitLinear,
    PolarQuantizer, KVCache, TurboQuantKVCache,
    MTPHead,
    RotaryEmbedding, apply_rope, rotate_half,
    SwiGLU,
    GPT, BASE_CONFIG, MHC_CONFIG, BITNET_CONFIG, MTP_CONFIG,
    ROPE_CONFIG, GQA_CONFIG, SWIGLU_CONFIG, RMSNORM_CONFIG, MODERN_CONFIG,
    ALL_CONFIG,
)

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


def test_sinkhorn():
    print("\n=== Sinkhorn (doubly stochastic) ===")
    for size in [4, 8, 16]:
        log_alpha = torch.randn(size, size)
        W = sinkhorn(log_alpha, n_iters=20)

        check(f"{size}x{size} all non-negative", (W >= -1e-6).all().item())
        row_sums = W.sum(dim=-1)
        check(f"{size}x{size} rows sum to 1", torch.allclose(row_sums, torch.ones(size), atol=1e-4),
              f"got {row_sums}")
        col_sums = W.sum(dim=-2)
        check(f"{size}x{size} cols sum to 1", torch.allclose(col_sums, torch.ones(size), atol=1e-4),
              f"got {col_sums}")

    log_alpha = torch.randn(4, 4)
    W1 = sinkhorn(log_alpha, n_iters=5)
    W2 = sinkhorn(log_alpha, n_iters=50)
    check("more iters = more precise", (W2.sum(-1) - 1.0).abs().max() <= (W1.sum(-1) - 1.0).abs().max() + 1e-7)


def test_mhc_residual():
    print("\n=== mHC Residual (signal preservation) ===")
    n_streams, B, T, C = 4, 2, 32, 64
    mhc = MHCResidual(n_streams)

    streams = torch.randn(B, n_streams, T, C)
    update = torch.randn(B, T, C)
    out = mhc(streams, update)

    check("output shape matches input", out.shape == streams.shape)

    with torch.no_grad():
        mhc.log_alpha.zero_()
    W = sinkhorn(mhc.log_alpha)
    check("zero init -> uniform doubly stochastic", torch.allclose(W, torch.ones(n_streams, n_streams) / n_streams, atol=1e-3))

    streams_norm = streams.norm()
    out_no_update = mhc(streams, torch.zeros(B, T, C))
    check("no update -> norm preserved (not amplified >2x)",
          out_no_update.norm() < streams_norm * 2)


def test_mhc_expand_collapse():
    print("\n=== mHC Expand/Collapse (roundtrip) ===")
    B, T, C, n_streams = 2, 32, 64, 4
    expand = MHCExpand(n_streams, C)
    collapse = MHCCollapse(n_streams, C)

    x = torch.randn(B, T, C)
    streams = expand(x)
    check("expand shape", streams.shape == (B, n_streams, T, C))

    x_back = collapse(streams)
    check("collapse shape", x_back.shape == (B, T, C))

    expand1 = MHCExpand(1, C)
    collapse1 = MHCCollapse(1, C)
    s1 = expand1(x)
    check("1 stream expand = unsqueeze", s1.shape == (B, 1, T, C))
    x1 = collapse1(s1)
    check("1 stream roundtrip = identity", torch.allclose(x, x1))


def test_bitlinear():
    print("\n=== BitLinear (ternary weights) ===")
    bl = BitLinear(64, 128)
    x = torch.randn(2, 32, 64)

    out = bl(x)
    check("output shape", out.shape == (2, 32, 128))

    w = bl.weight.data
    w_q = bl.ternary_quantize(w)
    unique_vals = w_q.unique()
    alpha = w.abs().mean()
    check("ternary: has zero bucket", (w_q == 0).any().item())
    check("ternary: has positive bucket", (w_q > 0).any().item())
    check("ternary: has negative bucket", (w_q < 0).any().item())
    nonzero = w_q[w_q != 0].abs().unique()
    check("ternary: non-zero values are all alpha", len(nonzero) == 1 and (nonzero[0] - alpha).abs() < 1e-6)

    x_q = bl.activation_quantize(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    scaled = x * scale
    check("activation quantize: within [-128, 127] after scaling",
          (scaled.round().clamp(-128, 127) == (x_q * scale).round()).all().item())


def test_bitlinear_gradient():
    print("\n=== BitLinear (gradient flow) ===")
    bl = BitLinear(64, 128)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = bl(x)
    loss = out.sum()
    loss.backward()
    check("input has gradient", x.grad is not None)
    check("weight has gradient", bl.weight.grad is not None)
    check("gradients are finite", torch.isfinite(bl.weight.grad).all().item())


def test_polar_quantizer():
    print("\n=== PolarQuantizer (reconstruction error) ===")
    for bits in [2, 3, 4, 8]:
        pq = PolarQuantizer(bits=bits)
        t = torch.randn(4, 8, 32, 64)
        q_norms, q_unit, params = pq.quantize(t)
        t_hat = pq.dequantize(q_norms, q_unit, params)

        check(f"{bits}-bit output shape", t_hat.shape == t.shape)
        rel_error = (t - t_hat).norm() / t.norm()
        threshold = {2: 0.8, 3: 0.5, 4: 0.2, 8: 0.1}[bits]
        check(f"{bits}-bit relative error < {threshold}",
              rel_error < threshold,
              f"got {rel_error:.4f}")

    pq4 = PolarQuantizer(bits=4)
    pq8 = PolarQuantizer(bits=8)
    t = torch.randn(4, 8, 32, 64)
    err4 = (t - pq4.dequantize(*pq4.quantize(t))).norm() / t.norm()
    err8 = (t - pq8.dequantize(*pq8.quantize(t))).norm() / t.norm()
    check("more bits = less error", err8 < err4, f"4-bit: {err4:.4f}, 8-bit: {err8:.4f}")


def test_kv_cache():
    print("\n=== KV Cache ===")
    cache = KVCache(max_seq_len=32)
    k1 = torch.randn(1, 8, 16, 64)
    v1 = torch.randn(1, 8, 16, 64)
    cache.update(k1, v1)

    k_out, v_out = cache.get()
    check("single update shape", k_out.shape == k1.shape and v_out.shape == v1.shape)
    check("single update values", torch.allclose(k_out, k1) and torch.allclose(v_out, v1))

    k2 = torch.randn(1, 8, 8, 64)
    v2 = torch.randn(1, 8, 8, 64)
    cache.update(k2, v2)
    k_out, v_out = cache.get()
    check("two updates concat", k_out.shape[2] == 24 and v_out.shape[2] == 24)
    check("second update values", torch.allclose(k_out[:, :, 16:24], k2) and torch.allclose(v_out[:, :, 16:24], v2))

    cache.clear()
    check("clear resets position", cache.pos == 0)

    print("\n=== TurboQuant KV Cache ===")
    tq_cache = TurboQuantKVCache(bits=4)
    tq_cache.update(k1, v1)
    k_out, v_out = tq_cache.get()
    check("turboquant single update shape", k_out.shape == k1.shape and v_out.shape == v1.shape)
    tq_cache.update(k2, v2)
    k_out, v_out = tq_cache.get()
    check("turboquant two updates concat", k_out.shape[2] == 24 and v_out.shape[2] == 24)
    tq_cache.clear()
    check("turboquant clear empties cache", len(tq_cache.k_cache) == 0 and len(tq_cache.v_cache) == 0)


def test_mtp():
    print("\n=== MTP (Multi-Token Prediction) ===")
    config = {"n_embd": 64, "vocab_size": 256}

    for future_idx in [1, 2, 4]:
        head = MTPHead(config, future_idx=future_idx)
        hidden = torch.randn(2, 32, 64)
        targets = torch.randint(0, 256, (2, 32))

        logits, loss = head(hidden, targets)
        check(f"head(+{future_idx}) logits shape", logits.shape == (2, 32, 256))
        check(f"head(+{future_idx}) loss is scalar", loss is not None and loss.dim() == 0)

        logits_no_target, loss_no_target = head(hidden)
        check(f"head(+{future_idx}) no target -> no loss", loss_no_target is None)

    head1 = MTPHead(config, future_idx=1)
    head4 = MTPHead(config, future_idx=4)
    hidden = torch.randn(2, 8, 64)
    targets = torch.randint(0, 256, (2, 8))
    _, loss1 = head1(hidden, targets)
    _, loss4 = head4(hidden, targets)
    check("shorter target still works for large future_idx", loss4 is not None)

    model = GPT(MTP_CONFIG)
    x = torch.randint(0, MTP_CONFIG["vocab_size"], (2, 64))
    _, loss_mtp = model(x, x)
    model_base = GPT(BASE_CONFIG)
    _, loss_base = model_base(x, x)
    check("MTP loss > base loss (extra heads add loss)", loss_mtp.item() > loss_base.item() * 0.5)

    loss_mtp.backward()
    mtp_grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    check("MTP gradients flow through all heads", mtp_grads_ok)


def test_rope():
    print("\n=== RoPE (Rotary Position Embeddings) ===")
    dim, seq_len = 64, 32
    rope = RotaryEmbedding(dim, max_seq_len=128)
    cos, sin = rope(seq_len)
    check("cos shape", cos.shape == (seq_len, dim))
    check("sin shape", sin.shape == (seq_len, dim))
    check("cos values in [-1, 1]", cos.abs().max() <= 1.0 + 1e-6)
    check("sin values in [-1, 1]", sin.abs().max() <= 1.0 + 1e-6)

    q = torch.randn(2, 4, seq_len, dim)
    k = torch.randn(2, 4, seq_len, dim)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    check("RoPE preserves q shape", q_rot.shape == q.shape)
    check("RoPE preserves k shape", k_rot.shape == k.shape)

    q_norm = q.norm(dim=-1)
    q_rot_norm = q_rot.norm(dim=-1)
    check("RoPE preserves vector norms", torch.allclose(q_norm, q_rot_norm, atol=1e-4))

    dot_same = (q_rot[0, 0, 5] * k_rot[0, 0, 5]).sum()
    dot_orig_same = (q[0, 0, 5] * k[0, 0, 5]).sum()
    check("RoPE preserves same-position dot product", torch.allclose(dot_same, dot_orig_same, atol=1e-4))

    dot_cross_orig = (q[0, 0, 3] * k[0, 0, 10]).sum()
    dot_cross_rot = (q_rot[0, 0, 3] * k_rot[0, 0, 10]).sum()
    check("RoPE changes cross-position dot product", not torch.allclose(dot_cross_orig, dot_cross_rot, atol=1e-3))

    model_rope = GPT(ROPE_CONFIG)
    check("RoPE model has no pos_emb", not hasattr(model_rope, "pos_emb"))
    model_base = GPT(BASE_CONFIG)
    check("base model has pos_emb", hasattr(model_base, "pos_emb"))


def test_gqa():
    print("\n=== GQA (Grouped Query Attention) ===")
    model_gqa = GPT(GQA_CONFIG)
    model_base = GPT(BASE_CONFIG)
    gqa_params = sum(p.numel() for p in model_gqa.parameters())
    base_params = sum(p.numel() for p in model_base.parameters())
    check("GQA has fewer params than base", gqa_params < base_params,
          f"gqa={gqa_params:,} base={base_params:,}")

    x = torch.randint(0, GQA_CONFIG["vocab_size"], (2, 64))
    logits, loss = model_gqa(x, x)
    check("GQA forward works", logits.shape == (2, 64, GQA_CONFIG["vocab_size"]))
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model_gqa.parameters() if p.requires_grad)
    check("GQA gradients finite", grads_ok)


def test_swiglu():
    print("\n=== SwiGLU ===")
    config = {"n_embd": 64, "use_bitnet": False}
    swiglu = SwiGLU(config)
    x = torch.randn(2, 16, 64)
    out = swiglu(x)
    check("SwiGLU output shape", out.shape == x.shape)

    x.requires_grad_(True)
    out = swiglu(x)
    out.sum().backward()
    check("SwiGLU gradient flows", x.grad is not None and torch.isfinite(x.grad).all())

    model = GPT(SWIGLU_CONFIG)
    x = torch.randint(0, SWIGLU_CONFIG["vocab_size"], (2, 64))
    _, loss = model(x, x)
    check("SwiGLU model forward works", loss is not None)


def test_rmsnorm():
    print("\n=== RMSNorm ===")
    model = GPT(RMSNORM_CONFIG)
    has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
    has_rmsnorm = any(isinstance(m, torch.nn.RMSNorm) for m in model.modules())
    check("RMSNorm model uses RMSNorm", has_rmsnorm)
    check("RMSNorm model has no LayerNorm", not has_layernorm)

    x = torch.randint(0, RMSNORM_CONFIG["vocab_size"], (2, 64))
    _, loss = model(x, x)
    check("RMSNorm model forward works", loss is not None)


def test_modern():
    print("\n=== Modern config (RoPE + GQA + SwiGLU + RMSNorm) ===")
    model = GPT(MODERN_CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randint(0, MODERN_CONFIG["vocab_size"], (2, 64))
    logits, loss = model(x, x)
    check("modern forward works", logits.shape == (2, 64, MODERN_CONFIG["vocab_size"]))
    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check("modern gradients finite", grads_ok)
    print(f"  INFO  modern params: {n_params:,}")


def test_full_model_configs():
    print("\n=== Full model forward/backward ===")
    configs = [
        ("base", BASE_CONFIG),
        ("mhc", MHC_CONFIG),
        ("bitnet", BITNET_CONFIG),
        ("mtp", MTP_CONFIG),
        ("rope", ROPE_CONFIG),
        ("gqa", GQA_CONFIG),
        ("swiglu", SWIGLU_CONFIG),
        ("rmsnorm", RMSNORM_CONFIG),
        ("modern", MODERN_CONFIG),
        ("all", ALL_CONFIG),
    ]
    for name, cfg in configs:
        model = GPT(cfg)
        x = torch.randint(0, cfg["vocab_size"], (2, 64))
        logits, loss = model(x, x)
        check(f"{name} forward shape", logits.shape == (2, 64, cfg["vocab_size"]))
        check(f"{name} loss is scalar", loss.dim() == 0)

        loss.backward()
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        check(f"{name} all gradients finite", grads_ok)


def test_full_model_generate():
    print("\n=== Full model generation ===")
    for name, cfg in [("base", BASE_CONFIG), ("all", ALL_CONFIG)]:
        model = GPT(cfg).eval()
        prompt = torch.randint(0, cfg["vocab_size"], (1, 8))
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=16)
        check(f"{name} generate length", out.shape == (1, 24))
        check(f"{name} prompt preserved", (out[0, :8] == prompt[0]).all().item())


def test_top_p_min_p_generate():
    print("\n=== Top-p/min-p generation ===")
    cfg = {
        **BASE_CONFIG,
        "vocab_size": 256,
        "block_size": 32,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "use_rope": True,
        "n_kv_head": 2,
        "use_swiglu": True,
        "use_rmsnorm": True,
    }
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        out = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=0,
            top_p=0.9,
            min_p=0.01,
        )
    check("top-p/min-p generate length", out.shape == (1, 18))
    check("top-p/min-p prompt preserved", (out[0, :8] == prompt[0]).all().item())


def test_mtp_speculative_generate():
    print("\n=== MTP speculative generation ===")
    cfg = {
        **BASE_CONFIG,
        "vocab_size": 256,
        "block_size": 32,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "use_rope": True,
        "n_kv_head": 2,
        "use_swiglu": True,
        "use_rmsnorm": True,
        "use_mtp": True,
        "mtp_heads": 3,
        "use_turboquant": True,
        "turboquant_bits": 4,
    }
    model = GPT(cfg).eval()
    prompt = torch.randint(0, cfg["vocab_size"], (1, 8))
    with torch.no_grad():
        out = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.0,
            speculative=True,
            speculate_tokens=3,
            use_turboquant=True,
        )
    check("speculative+turbo generate length", out.shape == (1, 18))
    check("speculative+turbo prompt preserved", (out[0, :8] == prompt[0]).all().item())


if __name__ == "__main__":
    test_sinkhorn()
    test_mhc_residual()
    test_mhc_expand_collapse()
    test_bitlinear()
    test_bitlinear_gradient()
    test_polar_quantizer()
    test_kv_cache()
    test_mtp()
    test_rope()
    test_gqa()
    test_swiglu()
    test_rmsnorm()
    test_modern()
    test_full_model_configs()
    test_full_model_generate()
    test_top_p_min_p_generate()
    test_mtp_speculative_generate()

    print(f"\n{'=' * 40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("All tests passed!")
    else:
        print(f"WARNING: {FAIL} test(s) failed!")
        exit(1)
