import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# --- mHC: Manifold-Constrained Hyper-Connections ---

def sinkhorn(log_alpha, n_iters=5):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return log_alpha.exp()


class MHCResidual(nn.Module):
    def __init__(self, n_streams):
        super().__init__()
        self.n_streams = n_streams
        self.log_alpha = nn.Parameter(torch.zeros(n_streams, n_streams))

    def forward(self, streams, update):
        W = sinkhorn(self.log_alpha)
        mixed = torch.einsum("ij,bjte->bite", W, streams)
        mixed[:, 0] = mixed[:, 0] + update
        return mixed


class MHCExpand(nn.Module):
    def __init__(self, n_streams, n_embd):
        super().__init__()
        self.n_streams = n_streams
        self.proj = nn.Linear(n_embd, n_streams * n_embd) if n_streams > 1 else None

    def forward(self, x):
        if self.n_streams == 1:
            return x.unsqueeze(1)
        B, T, C = x.shape
        return self.proj(x).view(B, self.n_streams, T, C)


class MHCCollapse(nn.Module):
    def __init__(self, n_streams, n_embd):
        super().__init__()
        self.n_streams = n_streams
        self.proj = nn.Linear(n_streams * n_embd, n_embd) if n_streams > 1 else None

    def forward(self, streams):
        if self.n_streams == 1:
            return streams.squeeze(1)
        B, S, T, C = streams.shape
        return self.proj(streams.permute(0, 2, 1, 3).reshape(B, T, S * C))


# --- BitNet: Ternary weight linear layer ---

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rms_norm = nn.RMSNorm(in_features)
        nn.init.normal_(self.weight, std=0.02)

    def ternary_quantize(self, w):
        alpha = w.abs().mean()
        threshold = alpha * 0.5
        w_ternary = torch.zeros_like(w)
        w_ternary[w > threshold] = alpha
        w_ternary[w < -threshold] = -alpha
        return w + (w_ternary - w).detach()

    def activation_quantize(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x * scale
        x_q = x_scaled + (x_scaled.round().clamp(-128, 127) - x_scaled).detach()
        return x_q / scale

    def forward(self, x):
        x = self.rms_norm(x)
        w_q = self.ternary_quantize(self.weight)
        x_q = self.activation_quantize(x)
        out = F.linear(x_q, w_q, self.bias)
        return out


def make_linear(in_f, out_f, bias=True, use_bitnet=False):
    if use_bitnet:
        return BitLinear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


# --- TurboQuant: KV-cache compression for inference ---

class PolarQuantizer:
    def __init__(self, bits=4):
        self.bits = bits
        self.levels = 2 ** bits

    def quantize(self, tensor):
        norms = tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = tensor / norms
        norm_min = norms.min()
        norm_max = norms.max()
        norm_scale = (norm_max - norm_min) / (self.levels - 1)
        q_norms = ((norms - norm_min) / norm_scale.clamp(min=1e-8)).round().clamp(0, self.levels - 1)
        val_min = unit.min()
        val_max = unit.max()
        val_scale = (val_max - val_min) / (self.levels - 1)
        q_unit = ((unit - val_min) / val_scale.clamp(min=1e-8)).round().clamp(0, self.levels - 1)
        return q_norms, q_unit, (norm_min, norm_scale, val_min, val_scale)

    def dequantize(self, q_norms, q_unit, params):
        norm_min, norm_scale, val_min, val_scale = params
        norms = q_norms * norm_scale + norm_min
        unit = q_unit * val_scale + val_min
        return unit * norms


class TurboQuantKVCache:
    def __init__(self, bits=4):
        self.quantizer = PolarQuantizer(bits=bits)
        self.k_cache = []
        self.v_cache = []

    def update(self, k_new, v_new):
        qk_norms, qk_unit, k_params = self.quantizer.quantize(k_new)
        qv_norms, qv_unit, v_params = self.quantizer.quantize(v_new)
        self.k_cache.append((qk_norms, qk_unit, k_params))
        self.v_cache.append((qv_norms, qv_unit, v_params))

    def get(self):
        ks = [self.quantizer.dequantize(*entry) for entry in self.k_cache]
        vs = [self.quantizer.dequantize(*entry) for entry in self.v_cache]
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    def clear(self):
        self.k_cache.clear()
        self.v_cache.clear()


# --- MTP: Multi-Token Prediction ---

class MTPHead(nn.Module):
    def __init__(self, config, future_idx):
        super().__init__()
        self.future_idx = future_idx
        n_embd = config["n_embd"]
        vocab_size = config["vocab_size"]
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, hidden, targets=None):
        h = self.ln(self.proj(hidden))
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            shift = self.future_idx
            if targets.size(1) > shift:
                logits_shifted = logits[:, :-shift].contiguous()
                targets_shifted = targets[:, shift:].contiguous()
                loss = F.cross_entropy(
                    logits_shifted.view(-1, logits_shifted.size(-1)),
                    targets_shifted.view(-1),
                    ignore_index=-1,
                )
        return logits, loss


# --- RoPE: Rotary Position Embeddings ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# --- SwiGLU MLP ---

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config["n_embd"]
        hidden = int(4 * n_embd * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        use_bitnet = config.get("use_bitnet", False)
        self.gate = make_linear(n_embd, hidden, bias=False, use_bitnet=use_bitnet)
        self.up = make_linear(n_embd, hidden, bias=False, use_bitnet=use_bitnet)
        self.down = make_linear(hidden, n_embd, bias=False, use_bitnet=use_bitnet)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# --- Core model ---

def make_norm(n_embd, use_rmsnorm=False):
    if use_rmsnorm:
        return nn.RMSNorm(n_embd)
    return nn.LayerNorm(n_embd)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.n_kv_head = config.get("n_kv_head", self.n_head)
        self.head_dim = self.n_embd // self.n_head
        self.use_rope = config.get("use_rope", False)
        use_bitnet = config.get("use_bitnet", False)

        self.q_proj = make_linear(self.n_embd, self.n_head * self.head_dim, use_bitnet=use_bitnet)
        self.k_proj = make_linear(self.n_embd, self.n_kv_head * self.head_dim, use_bitnet=use_bitnet)
        self.v_proj = make_linear(self.n_embd, self.n_kv_head * self.head_dim, use_bitnet=use_bitnet)
        self.proj = make_linear(self.n_embd, self.n_embd, use_bitnet=use_bitnet)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=config.get("block_size", 512))

    def forward(self, x, kv_cache=None, pos_offset=0):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(pos_offset + T)
            cos, sin = cos[pos_offset:pos_offset + T], sin[pos_offset:pos_offset + T]
            q, k = apply_rope(q, k, cos, sin)

        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        use_causal = (T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        use_bitnet = config.get("use_bitnet", False)
        self.fc = make_linear(config["n_embd"], 4 * config["n_embd"], use_bitnet=use_bitnet)
        self.proj = make_linear(4 * config["n_embd"], config["n_embd"], use_bitnet=use_bitnet)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.use_mhc = config.get("use_mhc", False)
        use_rmsnorm = config.get("use_rmsnorm", False)
        self.ln1 = make_norm(config["n_embd"], use_rmsnorm)
        self.attn = CausalSelfAttention(config)
        self.ln2 = make_norm(config["n_embd"], use_rmsnorm)
        if config.get("use_swiglu", False):
            self.mlp = SwiGLU(config)
        else:
            self.mlp = MLP(config)
        if self.use_mhc:
            n_streams = config.get("mhc_streams", 4)
            self.mhc_attn = MHCResidual(n_streams)
            self.mhc_mlp = MHCResidual(n_streams)

    def forward(self, x, streams=None, kv_cache=None, pos_offset=0):
        if self.use_mhc and streams is not None:
            inp = streams[:, 0]
            attn_out = self.attn(self.ln1(inp), kv_cache=kv_cache, pos_offset=pos_offset)
            streams = self.mhc_attn(streams, attn_out)
            mlp_inp = streams[:, 0]
            mlp_out = self.mlp(self.ln2(mlp_inp))
            streams = self.mhc_mlp(streams, mlp_out)
            return streams
        else:
            x = x + self.attn(self.ln1(x), kv_cache=kv_cache, pos_offset=pos_offset)
            x = x + self.mlp(self.ln2(x))
            return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_mhc = config.get("use_mhc", False)
        self.use_mtp = config.get("use_mtp", False)
        self.use_rope = config.get("use_rope", False)
        self.mtp_heads_n = config.get("mtp_heads", 4)
        self.mtp_weight = config.get("mtp_weight", 0.1)
        self.use_turboquant = config.get("use_turboquant", False)
        self.turboquant_bits = config.get("turboquant_bits", 4)
        use_rmsnorm = config.get("use_rmsnorm", False)

        self.tok_emb = nn.Embedding(config["vocab_size"], config["n_embd"])
        if not self.use_rope:
            self.pos_emb = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config["n_layer"])])
        self.ln_f = make_norm(config["n_embd"], use_rmsnorm)
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        self.tok_emb.weight = self.lm_head.weight

        if self.use_mhc:
            n_streams = config.get("mhc_streams", 4)
            self.mhc_expand = MHCExpand(n_streams, config["n_embd"])
            self.mhc_collapse = MHCCollapse(n_streams, config["n_embd"])

        if self.use_mtp:
            self.mtp_heads = nn.ModuleList([
                MTPHead(config, future_idx=i + 1) for i in range(self.mtp_heads_n)
            ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config["block_size"]:
            raise ValueError(f"Input length {T} exceeds block_size {self.config['block_size']}")
        x = self.tok_emb(idx)
        if not self.use_rope:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)

        if self.use_mhc:
            streams = self.mhc_expand(x)
            for block in self.blocks:
                streams = block(x, streams=streams)
            x = self.mhc_collapse(streams)
        else:
            for block in self.blocks:
                x = block(x)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.use_mtp:
                for head in self.mtp_heads:
                    _, mtp_loss = head(hidden, targets)
                    if mtp_loss is not None:
                        loss = loss + self.mtp_weight * mtp_loss
        return logits, loss

    def _forward_inference(self, x, kv_caches, pos_offset=0):
        if self.use_mhc:
            streams = self.mhc_expand(x)
            for block, cache in zip(self.blocks, kv_caches or [None] * len(self.blocks)):
                streams = block(x, streams=streams, kv_cache=cache, pos_offset=pos_offset)
            x = self.mhc_collapse(streams)
        else:
            for block, cache in zip(self.blocks, kv_caches or [None] * len(self.blocks)):
                x = block(x, kv_cache=cache, pos_offset=pos_offset)
        return self.lm_head(self.ln_f(x))

    def _embed(self, tokens, pos_offset=0):
        x = self.tok_emb(tokens)
        if not self.use_rope:
            T = tokens.shape[1]
            pos = torch.arange(pos_offset, pos_offset + T, device=tokens.device)
            x = x + self.pos_emb(pos)
        return x

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        block_size = self.config["block_size"]
        idx = idx[:, -block_size:]
        has_cache = self.use_turboquant
        kv_caches = None
        if has_cache:
            kv_caches = [TurboQuantKVCache(bits=self.turboquant_bits) for _ in self.blocks]

        seq_len = idx.shape[1]
        x = self._embed(idx)
        logits = self._forward_inference(x, kv_caches, pos_offset=0)

        for i in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature
            if top_k is not None:
                k = min(top_k, logits_last.size(-1))
                v, _ = torch.topk(logits_last, k)
                logits_last[logits_last < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

            if i < max_new_tokens - 1:
                cur_pos = seq_len + i
                if has_cache and cur_pos < block_size:
                    x = self._embed(idx_next, pos_offset=cur_pos)
                    logits = self._forward_inference(x, kv_caches, pos_offset=cur_pos)
                else:
                    if kv_caches:
                        for c in kv_caches:
                            c.clear()
                    idx_cond = idx[:, -block_size:]
                    x = self._embed(idx_cond)
                    logits = self._forward_inference(x, kv_caches, pos_offset=0)

        return idx


# --- Configs ---

BASE_CONFIG = {
    "vocab_size": 16384,
    "block_size": 512,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 12,
}

# Individual techniques
MHC_CONFIG = {**BASE_CONFIG, "use_mhc": True, "mhc_streams": 4}
BITNET_CONFIG = {**BASE_CONFIG, "use_bitnet": True}
MTP_CONFIG = {**BASE_CONFIG, "use_mtp": True, "mtp_heads": 4, "mtp_weight": 0.1}
ROPE_CONFIG = {**BASE_CONFIG, "use_rope": True}
GQA_CONFIG = {**BASE_CONFIG, "n_kv_head": 2}
SWIGLU_CONFIG = {**BASE_CONFIG, "use_swiglu": True}
RMSNORM_CONFIG = {**BASE_CONFIG, "use_rmsnorm": True}
TURBOQUANT_CONFIG = {**BASE_CONFIG, "use_turboquant": True, "turboquant_bits": 4}

# Combinations
MHC_BITNET_CONFIG = {**BASE_CONFIG, "use_mhc": True, "mhc_streams": 4, "use_bitnet": True}
MHC_MTP_CONFIG = {**BASE_CONFIG, "use_mhc": True, "mhc_streams": 4, "use_mtp": True, "mtp_heads": 4, "mtp_weight": 0.1}

# Modern LLaMA-style (RoPE + GQA + SwiGLU + RMSNorm)
MODERN_CONFIG = {**BASE_CONFIG, "use_rope": True, "n_kv_head": 2, "use_swiglu": True, "use_rmsnorm": True}

# Everything
ALL_CONFIG = {
    **BASE_CONFIG,
    "use_mhc": True, "mhc_streams": 4,
    "use_bitnet": True,
    "use_mtp": True, "mtp_heads": 4, "mtp_weight": 0.1,
    "use_rope": True, "n_kv_head": 2,
    "use_swiglu": True, "use_rmsnorm": True,
    "use_turboquant": True, "turboquant_bits": 4,
}

RECOMMENDED_CONFIG = {
    **BASE_CONFIG,
    "use_rope": True, "n_kv_head": 2,
    "use_swiglu": True, "use_rmsnorm": True,
    "use_mtp": True, "mtp_heads": 4, "mtp_weight": 0.1,
}

MODEL_CONFIG = RECOMMENDED_CONFIG

if __name__ == "__main__":
    configs = {
        "base": BASE_CONFIG,
        "rope": ROPE_CONFIG,
        "gqa": GQA_CONFIG,
        "swiglu": SWIGLU_CONFIG,
        "rmsnorm": RMSNORM_CONFIG,
        "modern": MODERN_CONFIG,
        "mhc": MHC_CONFIG,
        "bitnet": BITNET_CONFIG,
        "mtp": MTP_CONFIG,
        "all": ALL_CONFIG,
    }
    for name, cfg in configs.items():
        model = GPT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, cfg["vocab_size"], (2, 64))
        logits, loss = model(x, x)
        print(f"{name:<12} | {n_params:>12,} params ({n_params/1e6:.1f}M) | loss: {loss.item():.2f}")
