# Train a Language Model from Scratch

A from-scratch GPT-style language model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), featuring modern architecture techniques from LLaMA, DeepSeek-V3, and Google Research. Built to run on consumer GPUs.

## Sample Output

```
Prompt: Once upon a time,
Output: Once upon a time, there was a little girl named Lily. She loved to play
dress-up and play with her toys. One day, Lily's mommy took her to the store
to buy a new dress. Lily's mommy bought it for her and Lily was so happy...
```

Trained on an RTX 2060 Super; generates at **~160 tok/s** with `torch.compile` (dynamic-shape) decoding, vs **~90 tok/s** eager. A trained 19M checkpoint is on the Hub: [**epoyraz/tinystories-25m**](https://huggingface.co/epoyraz/tinystories-25m).

## Architecture

Decoder-only transformer with independently configurable techniques:

| Technique | Origin | What it does | Config Flag |
|---|---|---|---|
| **RoPE** | LLaMA | Rotary position embeddings — better length generalization | `use_rope: True` |
| **GQA** | LLaMA 2 | Grouped Query Attention — shared KV heads, less VRAM | `n_kv_head: 2` |
| **SwiGLU** | LLaMA | Gated MLP activation — better quality at same param count | `use_swiglu: True` |
| **RMSNorm** | LLaMA | Simpler, faster normalization layer | `use_rmsnorm: True` |
| **QK-Norm** | modded-nanoGPT | RMSNorm on Q and K before attention — stabilizes training, helps convergence | `use_qk_norm: True` |
| **ReLU²** | modded-nanoGPT | Ungated MLP with squared-ReLU activation — simpler alternative to SwiGLU | `use_relu2: True` |
| **Logit soft-cap** | Gemma 2 | `cap·tanh(logits/cap)` — bounds logits for stability | `logit_cap: 15.0` |
| **Zero-init** | modded-nanoGPT | Zero-init block output projections (muP-like) — each block starts as identity | `use_zero_init: True` |
| **MTP** | DeepSeek-V3 | Multi-Token Prediction — better sample efficiency + speculative decoding | `use_mtp: True` |
| **mHC** | DeepSeek | Manifold-Constrained Hyper-Connections — learned residual routing | `use_mhc: True` |
| **BitNet** | Microsoft | Ternary 1-bit weights with straight-through estimator | `use_bitnet: True` |
| **FastBitNet** | Microsoft | BitNet with INT8 tensor-core matmuls at inference (`torch._int_mm`) | `use_fast_bitnet: True` |
| **TurboQuant** | Google | KV-cache compression via PolarQuant + QJL | `use_turboquant: True` |

`use_fast_bitnet` trains with straight-through ternary (like BitNet) but runs inference
through INT8 tensor cores; such checkpoints can be packed to ~2 bits/weight with
`export_ternary.py` (8× smaller on disk).

## What each optimization buys you

All measured on an RTX 2060 Super. Each lever acts on a different axis — train speed,
inference speed, convergence, or memory — so they stack:

| Lever | Train tok/s | Infer tok/s | Loss per step | Train memory |
|---|---|---|---|---|
| **`torch.compile`** | **1.4–1.5×** | **~1.8×** (dynamic) | — | — |
| **Muon** + QK-norm/ReLU²/softcap | ~same | — | **2.30 → 2.13** | — |
| **Zero-init** projections | — | — | **2.13 → 2.04** | — |
| **Chunked cross-entropy** | ↑ when memory-bound | — | — | **2.75× less** |
| **MTP** | slower | — | sample-eff. + spec-decode | more |

Notes: zero-init and chunked-CE are free (no quality change — verified by equivalence
tests). Muon converges much better per step but is ~slightly slower per step (≈even on the
modded recipe). Chunked-CE's *speed* win only shows up when you're memory-bound (e.g. large
batch without `torch.compile`); its **memory** saving is unconditional. MTP's heads are idle
during plain (non-speculative) decode, so they don't affect inference speed.

## Training Profiles

Pre-configured profiles optimized for different hardware:

Training throughput on an RTX 2060 Super (batch 32), eager vs. `--compile`:

| Profile | Params | Best for | tok/s eager | tok/s `--compile` |
|---|---|---|---|---|
| `tiny_fast` | 8.6M | Quick experiments | 173K | ~1.4× |
| `fast_2060` | 19M | RTX 2060 (no MTP) | 90K | **127K** |
| `fast_2060_mtp` | 19M | RTX 2060 with MTP | 58K | 89K |
| `fast_2060_mtp_fbitnet` | 19M | INT8 BitNet inference | 58K | — |
| `fast_2060_modded` | 19M | Best convergence (train with Muon) | 58K | — |
| `modern` | 42M | Larger GPUs (24GB+) | 48K | ~1.4× |
| `recommended` | 77M | Full featured | — | — |
| `base` | 46.5M | Vanilla GPT baseline | — | — |

`torch.compile` gives a measured **1.4–1.5×** speedup (see [torch.compile setup](#torchcompile-speedup-windows)).
MTP profiles are heavier (the extra prediction heads each run a full-vocab projection),
so they cap lower than the no-MTP profiles.

Find the optimal profile for your hardware:

```bash
python tune_training.py
python tune_training.py --profiles fast_2060_mtp modern recommended
```

## Optimizers

| Optimizer | Description |
|---|---|
| **AdamW** | Standard adaptive optimizer (default) |
| **Muon** | Newton-Schulz momentum for 2D+ weight matrices, paired with AdamW for embeddings/norms |

```bash
python train.py --profile fast_2060_modded --optimizer muon --max-lr 3e-3 --compile
```

### Measured convergence (RTX 2060 Super, `fast_2060`, 1,200 steps)

| Config | Optimizer | Val loss | tok/s |
|---|---|---|---|
| baseline | AdamW | 2.30 | 86K |
| + QK-Norm | AdamW | **2.27** | 76K |
| + ReLU² | AdamW | 2.31 | 72K |
| + logit-cap | AdamW | 2.33 | 74K |
| baseline | Muon | **2.14** | 41K |
| + all three | Muon | **2.13** | 35K |
| + all three + **zero-init** (`fast_2060_modded`) | Muon | **2.04** | 35K |

Takeaways from the A/B:
- **QK-Norm** is the one technique that helps under plain AdamW — keep it on.
- **ReLU²** and **logit soft-cap** only pay off paired with **Muon**'s higher LR (they slightly
  hurt under AdamW), which is why `fast_2060_modded` is meant to be trained with `--optimizer muon`.
- **Zero-init** (zeroing block output projections) is free (an init change) and lowered val
  loss 2.13 → 2.04 at equal steps — baked into the modded profiles.
- **Muon converges much better per step** (2.30 → 2.14) — but it's ~2× slower per step here
  (Newton-Schulz overhead), so at *equal wall-clock* AdamW trains further (AdamW reached **1.99**
  in the same ~240s that Muon used for 2.14). Muon is the choice when you're token/sample-limited
  or want the best loss-per-token; AdamW + QK-Norm + `--compile` is the fastest by the clock.

## Setup

```bash
# PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Dependencies
pip install datasets tokenizers trackio
```

## Quick Start

### 1. Download TinyStories dataset

```bash
python download_data.py
```

Downloads ~2.1M training examples (~1.8 GB) from HuggingFace.

### 2. Train a BPE tokenizer

```bash
python train_tokenizer.py
```

Trains a ByteLevel BPE tokenizer with 16K vocab. Takes ~2 minutes on CPU (runs in Rust via HuggingFace `tokenizers`).

### 3. Train the model

```bash
# Default (fast_2060 profile, AdamW)
python train.py

# With specific profile and optimizer
python train.py --profile fast_2060_mtp --batch-size 32 --max-steps 3000

# With Muon optimizer
python train.py --profile fast_2060_mtp --optimizer muon --max-lr 3e-3

# With activation checkpointing (saves ~40% VRAM)
python train.py --profile modern --activation-checkpointing

# With torch.compile (1.4-1.5x faster; see setup below)
python train.py --profile fast_2060 --compile

# All options
python train.py --help
```

#### torch.compile speedup (Windows)

`torch.compile` is the single biggest speed lever — **1.4–1.5× training, ~2× inference**,
and it's numerically lossless (verified: compiled and eager give identical validation
loss to ~1e-7). It needs a Triton backend plus a C compiler:

```bash
pip install triton-windows        # Triton build with Windows wheels (incl. cp314)
```

Triton also needs **MSVC** (`cl.exe`) to build its CUDA shims — the mingw `gcc` that's
often on PATH will fail. Install the **"Desktop development with C++"** workload from
[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/). You do **not**
need to configure PATH yourself: `msvc_env.py` locates and activates MSVC automatically
whenever you pass `--compile`. (On Linux/macOS, `pip install triton` + a system compiler
is all that's needed.)

### 4. Monitor training live

Open a second terminal:

```bash
python -c "import trackio; trackio.show(project='train-from-scratch')"
```

Opens a Gradio dashboard at `http://127.0.0.1:7860` with real-time loss curves, learning rate, and throughput graphs.

### 5. Generate text

```bash
# Default prompts + interactive REPL (type a prompt, "quit" to exit)
python generate.py

# Custom one-shot prompt
python generate.py --prompt "Once upon a time," --temperature 0.7

# Faster sustained/interactive decoding (~1.8x via torch.compile; one-time warmup)
python generate.py --compile --temperature 0.7

# Adjust sampling
python generate.py --temperature 0.6 --top-k 50 --max-tokens 300

# Greedy decoding
python generate.py --temperature 0

# All options
python generate.py --help
```

Defaults: `--max-tokens 200`, `--temperature 0.8`, `--top-k 40`, KV cache **on**.
`--compile` gives ~160 tok/s (vs ~90 eager) using dynamic-shape compilation, but pays a
one-time compile cost on the first generation — so it only helps for the REPL / many
generations; skip it for a single one-shot prompt.
`--speculative` (MTP draft + batched verify) and `--turboquant` (KV compression) are
implemented and correct, but both measured **slower** than plain compiled decode on a
19M model / RTX 2060, so they are opt-in and off by default.

### 6. Evaluate checkpoints

```bash
python evaluate.py
```

Runs all checkpoints against the validation set, prints loss and perplexity:

```
Checkpoint                         Val Loss   Perplexity
-------------------------------------------------------
step_1000.pt                         2.9417        18.95
step_2000.pt                         2.7185        15.16
final.pt                             2.6515        14.18
```

### 7. Benchmark configurations

```bash
python benchmark.py
```

Compares training speed, inference speed, and VRAM usage across all configurations.

### 8. Run correctness tests

```bash
python test_techniques.py
```

Verifies mathematical properties of each technique:
- **Sinkhorn**: doubly stochastic output (rows/cols sum to 1)
- **BitNet**: proper ternary quantization with zero bucket + STE gradients
- **PolarQuant**: bounded reconstruction error, monotonic with bit depth
- **RoPE**: norm preservation, position-dependent cross-attention
- **GQA**: fewer params than MHA, correct gradient flow
- **MTP**: shifted targets, auxiliary loss computation
- **KV Cache**: incremental updates, correct concatenation

## Training Results

On RTX 2060 Super (8GB VRAM), `fast_2060_modded` + Muon + `--compile`, batch 40, 3,000
steps — the run that produced [epoyraz/tinystories-25m](https://huggingface.co/epoyraz/tinystories-25m):

| Recipe | Steps | Val Loss | Time |
|---|---|---|---|
| `fast_2060_mtp` + AdamW (baseline) | 3,000 | 2.65 | ~7 min |
| `fast_2060_modded` + Muon + zero-init | 3,000 | **2.40** | ~8 min |

Loss is the combined objective (next-token cross-entropy + `mtp_weight` × MTP auxiliary);
the pure next-token CE is lower. The modded recipe (QK-norm + ReLU² + logit-cap + zero-init,
trained with Muon) is a measured ~0.25 improvement over the AdamW baseline at equal steps.

## Project Structure

```
config.py              Shared device, dtype, and path configuration
model.py               GPT model with all configurable techniques
train.py               Training loop with CLI, profiles, Muon/AdamW, torch.compile, Trackio
train_tokenizer.py     BPE tokenizer training (16K vocab)
download_data.py       TinyStories dataset download from HuggingFace
generate.py            Text generation: KV cache, speculative decoding, --compile, REPL
evaluate.py            Validation loss and perplexity across checkpoints
benchmark.py           Speed and memory benchmarks across all configs
export_ternary.py      Pack FastBitLinear weights to ~2 bits/weight (8x smaller)
msvc_env.py            Auto-activates MSVC so torch.compile works on Windows
test_techniques.py     Correctness tests for all techniques (115 tests)
test_fast_bitnet.py    FastBitLinear INT8 correctness/speed/convergence tests (26 tests)
tune_training.py       Auto-detect optimal profile and batch size for your GPU
```

## Hardware Compatibility

| GPU | VRAM | Recommended Profile | Batch Size |
|---|---|---|---|
| RTX 2060 / 8GB | 8 GB | `fast_2060_mtp` | 32 |
| RTX 3070 / 8GB | 8 GB | `fast_2060_mtp` | 32 |
| RTX 3090 / 24GB | 24 GB | `modern` | 24-32 |
| RTX 4090 / 24GB | 24 GB | `recommended` | 40-64 |

Run `python tune_training.py` to find the best config for your specific hardware.

## Key Design Decisions

- **ByteLevel BPE** with 16K vocab — small enough to keep embedding tables light for sub-100M models
- **Weight tying** — token embeddings shared with output projection head
- **Cosine LR schedule** with linear warmup
- **fp16 mixed precision** with GradScaler
- **Incremental KV cache** for efficient autoregressive generation
- **Speculative decoding** via MTP heads — draft multiple tokens, verify in one forward pass

## References

- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (RoPE)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (SwiGLU)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (MTP, mHC)
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (BitNet)
- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
