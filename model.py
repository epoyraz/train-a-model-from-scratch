import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint


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
        return w_ternary.detach() + (w - w.detach())

    def activation_quantize(self, x):
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x * scale
        x_q = x_scaled.round().clamp(-128, 127).detach() + (x_scaled - x_scaled.detach())
        return x_q / scale

    def forward(self, x):
        x = self.rms_norm(x)
        w_q = self.ternary_quantize(self.weight)
        x_q = self.activation_quantize(x)
        out = F.linear(x_q, w_q, self.bias)
        return out


class FastBitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rms_norm = nn.RMSNorm(in_features)
        nn.init.normal_(self.weight, std=0.02)

    def _int8_forward(self, x):
        w = self.weight.detach()
        alpha = w.abs().mean()
        threshold = alpha * 0.5
        w_pos = (w > threshold).to(torch.int8)
        w_neg = (w < -threshold).to(torch.int8)

        x_max = x.detach().abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scale = 127.0 / x_max
        x_q = (x.detach() * x_scale).round().clamp(-128, 127).to(torch.int8)

        shape = x_q.shape
        x_2d = x_q.reshape(-1, shape[-1])

        rows = x_2d.shape[0]
        if rows <= 16:
            pad = 17 - rows
            x_2d = torch.nn.functional.pad(x_2d, (0, 0, 0, pad))
            y_pos = torch._int_mm(x_2d, w_pos.T)[:rows]
            y_neg = torch._int_mm(x_2d, w_neg.T)[:rows]
        else:
            y_pos = torch._int_mm(x_2d, w_pos.T)
            y_neg = torch._int_mm(x_2d, w_neg.T)

        y = (y_pos - y_neg).float().reshape(*shape[:-1], self.out_features)
        return y * (alpha / x_scale)

    def _ste_forward(self, x):
        alpha = self.weight.abs().mean()
        threshold = alpha * 0.5
        w_ternary = torch.zeros_like(self.weight)
        w_ternary[self.weight > threshold] = alpha
        w_ternary[self.weight < -threshold] = -alpha
        w_q = self.weight + (w_ternary - self.weight).detach()

        x_scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x * x_scale
        x_q = x_scaled + (x_scaled.round().clamp(-128, 127) - x_scaled).detach()
        x_q = x_q / x_scale

        return F.linear(x_q, w_q, None)

    def forward(self, x):
        x = self.rms_norm(x)
        if self.training:
            out = self._ste_forward(x)
        else:
            out = self._int8_forward(x)
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(in_f, out_f, bias=True, use_bitnet=False, use_fast_bitnet=False):
    if use_fast_bitnet:
        return FastBitLinear(in_f, out_f, bias=bias)
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
        unit = unit / unit.norm(dim=-1, keepdim=True).clamp(min=1e-8)
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


class KVCache:
    def __init__(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.k_cache = None
        self.v_cache = None
        self.pos = 0

    def _ensure_allocated(self, k_new, v_new):
        B, H, _, D = k_new.shape
        needs_alloc = (
            self.k_cache is None
            or self.k_cache.shape[0] != B
            or self.k_cache.shape[1] != H
            or self.k_cache.shape[3] != D
            or self.k_cache.device != k_new.device
            or self.k_cache.dtype != k_new.dtype
        )
        if needs_alloc:
            self.k_cache = torch.empty(
                B, H, self.max_seq_len, D,
                device=k_new.device,
                dtype=k_new.dtype,
            )
            self.v_cache = torch.empty(
                B, H, self.max_seq_len, D,
                device=v_new.device,
                dtype=v_new.dtype,
            )
            self.pos = 0

    def update(self, k_new, v_new):
        self._ensure_allocated(k_new, v_new)
        T = k_new.size(2)
        if self.pos + T > self.max_seq_len:
            raise ValueError(f"KV cache length {self.pos + T} exceeds max_seq_len {self.max_seq_len}")
        self.k_cache[:, :, self.pos:self.pos + T, :].copy_(k_new)
        self.v_cache[:, :, self.pos:self.pos + T, :].copy_(v_new)
        self.pos += T

    def get(self):
        if self.k_cache is None:
            return None, None
        return self.k_cache[:, :, :self.pos, :], self.v_cache[:, :, :self.pos, :]

    def clear(self):
        self.pos = 0


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
        use_fast_bitnet = config.get("use_fast_bitnet", False)
        self.gate = make_linear(n_embd, hidden, bias=False, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.up = make_linear(n_embd, hidden, bias=False, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.down = make_linear(hidden, n_embd, bias=False, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)

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
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})")
        self.head_dim = self.n_embd // self.n_head
        self.use_rope = config.get("use_rope", False)
        use_bitnet = config.get("use_bitnet", False)
        use_fast_bitnet = config.get("use_fast_bitnet", False)

        self.q_proj = make_linear(self.n_embd, self.n_head * self.head_dim, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.k_proj = make_linear(self.n_embd, self.n_kv_head * self.head_dim, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.v_proj = make_linear(self.n_embd, self.n_kv_head * self.head_dim, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.proj = make_linear(self.n_embd, self.n_embd, use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)

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
        use_fast_bitnet = config.get("use_fast_bitnet", False)
        self.fc = make_linear(config["n_embd"], 4 * config["n_embd"], use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)
        self.proj = make_linear(4 * config["n_embd"], config["n_embd"], use_bitnet=use_bitnet, use_fast_bitnet=use_fast_bitnet)

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
        self.use_activation_checkpointing = config.get("use_activation_checkpointing", False)
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
            if config.get("tie_mtp_lm_head", True):
                for head in self.mtp_heads:
                    head.lm_head.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_hidden(self, idx):
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
                if self.training and self.use_activation_checkpointing:
                    streams = checkpoint(lambda s, b=block: b(x, streams=s), streams, use_reentrant=False)
                else:
                    streams = block(x, streams=streams)
            x = self.mhc_collapse(streams)
        else:
            for block in self.blocks:
                if self.training and self.use_activation_checkpointing:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        return self.ln_f(x)

    def forward(self, idx, targets=None, return_hidden=False):
        hidden = self._compute_hidden(idx)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if self.use_mtp:
                for head in self.mtp_heads:
                    _, mtp_loss = head(hidden, targets)
                    if mtp_loss is not None:
                        loss = loss + self.mtp_weight * mtp_loss
        if return_hidden:
            return logits, loss, hidden
        return logits, loss

    def _forward_inference(self, x, kv_caches, pos_offset=0, return_hidden=False):
        if self.use_mhc:
            streams = self.mhc_expand(x)
            for block, cache in zip(self.blocks, kv_caches or [None] * len(self.blocks)):
                streams = block(x, streams=streams, kv_cache=cache, pos_offset=pos_offset)
            x = self.mhc_collapse(streams)
        else:
            for block, cache in zip(self.blocks, kv_caches or [None] * len(self.blocks)):
                x = block(x, kv_cache=cache, pos_offset=pos_offset)
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        if return_hidden:
            return logits, hidden
        return logits

    def _embed(self, tokens, pos_offset=0):
        x = self.tok_emb(tokens)
        if not self.use_rope:
            T = tokens.shape[1]
            pos = torch.arange(pos_offset, pos_offset + T, device=tokens.device)
            x = x + self.pos_emb(pos)
        return x

    def _filter_logits(self, logits, top_k=None, top_p=None, min_p=None):
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))

        if min_p is not None and min_p > 0:
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            remove = probs < (min_p * max_probs)
            top_token = logits.argmax(dim=-1, keepdim=True)
            remove.scatter_(dim=-1, index=top_token, value=False)
            logits = logits.masked_fill(remove, -float("inf"))

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = torch.zeros_like(logits, dtype=torch.bool)
            remove.scatter_(dim=-1, index=sorted_idx, src=sorted_remove)
            logits = logits.masked_fill(remove, -float("inf"))

        return logits

    def _distribution(self, logits, temperature=0.8, top_k=40, top_p=None, min_p=None):
        if temperature <= 0:
            token = logits.argmax(dim=-1, keepdim=True)
            probs = torch.zeros_like(logits)
            probs.scatter_(1, token, 1.0)
            return token, probs
        logits = self._filter_logits(logits / temperature, top_k=top_k, top_p=top_p, min_p=min_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return token, probs

    def _make_kv_caches(self, use_turboquant, use_kv_cache=True):
        if not use_kv_cache:
            return None
        if use_turboquant:
            return [TurboQuantKVCache(bits=self.turboquant_bits) for _ in self.blocks]
        return [KVCache(self.config["block_size"]) for _ in self.blocks]

    def _trim_or_seed_prompt(self, idx):
        block_size = self.config["block_size"]
        if idx.shape[1] == 0:
            eos_id = 1
            idx = torch.tensor([[eos_id]], dtype=idx.dtype, device=idx.device)
        return idx[:, -block_size:]

    def _prefill_generation(self, idx, use_turboquant=False, use_kv_cache=True):
        kv_caches = self._make_kv_caches(use_turboquant, use_kv_cache=use_kv_cache)
        seq_len = idx.shape[1]
        x = self._embed(idx)
        logits, hidden = self._forward_inference(x, kv_caches, pos_offset=0, return_hidden=True)
        return logits, hidden[:, -1:, :], kv_caches, seq_len

    def _advance_generation_state(self, idx, idx_next, kv_caches, seq_len, use_turboquant):
        block_size = self.config["block_size"]
        if kv_caches is not None and seq_len < block_size:
            x = self._embed(idx_next, pos_offset=seq_len)
            logits, hidden = self._forward_inference(x, kv_caches, pos_offset=seq_len, return_hidden=True)
            return logits, hidden[:, -1:, :], kv_caches, seq_len + 1

        use_kv_cache = kv_caches is not None
        if kv_caches:
            for cache in kv_caches:
                cache.clear()
        idx_cond = idx[:, -block_size:]
        logits, hidden, kv_caches, seq_len = self._prefill_generation(
            idx_cond,
            use_turboquant=use_turboquant,
            use_kv_cache=use_kv_cache,
        )
        return logits, hidden, kv_caches, seq_len

    def _generate_autoregressive(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=None,
        min_p=None,
        use_turboquant=None,
        use_kv_cache=True,
    ):
        idx = self._trim_or_seed_prompt(idx)
        use_turboquant = self.use_turboquant if use_turboquant is None else use_turboquant
        logits, last_hidden, kv_caches, seq_len = self._prefill_generation(
            idx,
            use_turboquant=use_turboquant,
            use_kv_cache=use_kv_cache,
        )

        for i in range(max_new_tokens):
            idx_next, _ = self._distribution(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            idx = torch.cat([idx, idx_next], dim=1)

            if i < max_new_tokens - 1:
                logits, last_hidden, kv_caches, seq_len = self._advance_generation_state(
                    idx, idx_next, kv_caches, seq_len, use_turboquant
                )

        return idx

    def _mtp_draft(self, last_hidden, n_tokens, temperature=0.8, top_k=40, top_p=None, min_p=None):
        draft_tokens = []
        draft_probs = []
        for head in self.mtp_heads[:n_tokens]:
            draft_logits, _ = head(last_hidden)
            token, probs = self._distribution(
                draft_logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            draft_tokens.append(token)
            draft_probs.append(probs)
        return draft_tokens, draft_probs

    def _mtp_speculative_generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=None,
        min_p=None,
        speculate_tokens=None,
        use_turboquant=None,
        use_kv_cache=True,
    ):
        if not self.use_mtp or idx.size(0) != 1:
            return self._generate_autoregressive(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                use_turboquant=use_turboquant,
                use_kv_cache=use_kv_cache,
            )

        idx = self._trim_or_seed_prompt(idx)
        use_turboquant = self.use_turboquant if use_turboquant is None else use_turboquant
        draft_width = speculate_tokens or self.mtp_heads_n
        draft_width = max(1, min(draft_width, self.mtp_heads_n))

        logits, last_hidden, kv_caches, seq_len = self._prefill_generation(
            idx,
            use_turboquant=use_turboquant,
            use_kv_cache=use_kv_cache,
        )
        generated = 0

        while generated < max_new_tokens:
            remaining = max_new_tokens - generated
            n_draft = min(draft_width, remaining)
            draft_tokens, draft_probs = self._mtp_draft(
                last_hidden,
                n_draft,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            accepted_all = True

            for draft_token, q_probs in zip(draft_tokens, draft_probs):
                target_token, p_probs = self._distribution(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )

                if temperature <= 0:
                    accept = torch.equal(draft_token, target_token)
                else:
                    proposed = draft_token.item()
                    p = p_probs[0, proposed]
                    q = q_probs[0, proposed].clamp(min=1e-12)
                    accept_prob = torch.minimum(torch.ones_like(p), p / q)
                    accept = torch.rand((), device=idx.device) <= accept_prob

                if accept:
                    idx_next = draft_token
                else:
                    accepted_all = False
                    if temperature <= 0:
                        idx_next = target_token
                    else:
                        residual = (p_probs - q_probs).clamp(min=0)
                        denom = residual.sum(dim=-1, keepdim=True)
                        if denom.item() <= 1e-12:
                            idx_next = target_token
                        else:
                            idx_next = torch.multinomial(residual / denom, num_samples=1)

                idx = torch.cat([idx, idx_next], dim=1)
                generated += 1
                if generated >= max_new_tokens:
                    break

                logits, last_hidden, kv_caches, seq_len = self._advance_generation_state(
                    idx, idx_next, kv_caches, seq_len, use_turboquant
                )

                if not accepted_all:
                    break

            if generated >= max_new_tokens:
                break

            if accepted_all:
                idx_next, _ = self._distribution(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                )
                idx = torch.cat([idx, idx_next], dim=1)
                generated += 1
                if generated < max_new_tokens:
                    logits, last_hidden, kv_caches, seq_len = self._advance_generation_state(
                        idx, idx_next, kv_caches, seq_len, use_turboquant
                    )

        return idx

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40,
        top_p=None,
        min_p=None,
        speculative=False,
        speculate_tokens=None,
        use_turboquant=None,
        use_kv_cache=True,
    ):
        if speculative:
            return self._mtp_speculative_generate(
                idx,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                speculate_tokens=speculate_tokens,
                use_turboquant=use_turboquant,
                use_kv_cache=use_kv_cache,
            )
        return self._generate_autoregressive(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            use_turboquant=use_turboquant,
            use_kv_cache=use_kv_cache,
        )


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
FAST_BITNET_CONFIG = {**BASE_CONFIG, "use_fast_bitnet": True}
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

FAST_2060_CONFIG = {
    **BASE_CONFIG,
    "block_size": 256,
    "n_embd": 384,
    "n_head": 6,
    "n_layer": 8,
    "use_rope": True,
    "n_kv_head": 2,
    "use_swiglu": True,
    "use_rmsnorm": True,
}

FAST_2060_MTP_CONFIG = {
    **FAST_2060_CONFIG,
    "use_mtp": True,
    "mtp_heads": 2,
    "mtp_weight": 0.1,
    "tie_mtp_lm_head": True,
}

FAST_2060_MTP_FBITNET_CONFIG = {
    **FAST_2060_MTP_CONFIG,
    "use_fast_bitnet": True,
}

FAST_2060_MTP_TURBO_CONFIG = {
    **FAST_2060_MTP_CONFIG,
    "use_turboquant": True,
    "turboquant_bits": 4,
}

TINY_FAST_CONFIG = {
    **BASE_CONFIG,
    "block_size": 256,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 6,
    "use_rope": True,
    "n_kv_head": 2,
    "use_swiglu": True,
    "use_rmsnorm": True,
}

LOW_MEMORY_2060_CONFIG = {
    **FAST_2060_CONFIG,
    "use_activation_checkpointing": True,
}

CONFIGS = {
    "base": BASE_CONFIG,
    "mhc": MHC_CONFIG,
    "bitnet": BITNET_CONFIG,
    "mtp": MTP_CONFIG,
    "rope": ROPE_CONFIG,
    "gqa": GQA_CONFIG,
    "swiglu": SWIGLU_CONFIG,
    "rmsnorm": RMSNORM_CONFIG,
    "turboquant": TURBOQUANT_CONFIG,
    "mhc_bitnet": MHC_BITNET_CONFIG,
    "mhc_mtp": MHC_MTP_CONFIG,
    "modern": MODERN_CONFIG,
    "all": ALL_CONFIG,
    "recommended": RECOMMENDED_CONFIG,
    "fast_2060": FAST_2060_CONFIG,
    "fast_2060_mtp": FAST_2060_MTP_CONFIG,
    "fast_2060_mtp_fbitnet": FAST_2060_MTP_FBITNET_CONFIG,
    "fast_2060_mtp_turbo": FAST_2060_MTP_TURBO_CONFIG,
    "tiny_fast": TINY_FAST_CONFIG,
    "low_memory_2060": LOW_MEMORY_2060_CONFIG,
}


def get_model_config(name="fast_2060", **overrides):
    if name not in CONFIGS:
        available = ", ".join(sorted(CONFIGS))
        raise ValueError(f"Unknown config '{name}'. Available configs: {available}")
    return {**CONFIGS[name], **{k: v for k, v in overrides.items() if v is not None}}


MODEL_CONFIG = RECOMMENDED_CONFIG

if __name__ == "__main__":
    configs = CONFIGS
    for name, cfg in configs.items():
        model = GPT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, cfg["vocab_size"], (2, 64))
        logits, loss = model(x, x)
        print(f"{name:<12} | {n_params:>12,} params ({n_params/1e6:.1f}M) | loss: {loss.item():.2f}")
