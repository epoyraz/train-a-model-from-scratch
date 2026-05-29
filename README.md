# Train a Language Model from Scratch

A from-scratch GPT-style language model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), featuring modern architecture techniques from LLaMA, DeepSeek-V3, and Google Research. Built to run on consumer GPUs.

## Sample Output

```
Prompt: Once upon a time,
Output: Once upon a time, there was a little girl named Lily. She loved to play
dress-up and play with her toys. One day, Lily's mommy took her to the store
to buy a new dress. Lily's mommy bought it for her and Lily was so happy...
```

Generated at **85 tok/s** with speculative decoding on an RTX 2060 Super.

## Architecture

Decoder-only transformer with 8 independently configurable techniques:

| Technique | Origin | What it does | Config Flag |
|---|---|---|---|
| **RoPE** | LLaMA | Rotary position embeddings — better length generalization | `use_rope: True` |
| **GQA** | LLaMA 2 | Grouped Query Attention — shared KV heads, less VRAM | `n_kv_head: 2` |
| **SwiGLU** | LLaMA | Gated MLP activation — better quality at same param count | `use_swiglu: True` |
| **RMSNorm** | LLaMA | Simpler, faster normalization layer | `use_rmsnorm: True` |
| **MTP** | DeepSeek-V3 | Multi-Token Prediction — better sample efficiency + speculative decoding | `use_mtp: True` |
| **mHC** | DeepSeek | Manifold-Constrained Hyper-Connections — learned residual routing | `use_mhc: True` |
| **BitNet** | Microsoft | Ternary 1-bit weights with straight-through estimator | `use_bitnet: True` |
| **TurboQuant** | Google | KV-cache compression via PolarQuant + QJL | `use_turboquant: True` |

## Training Profiles

Pre-configured profiles optimized for different hardware:

| Profile | Params | Best for | tok/s (2060) |
|---|---|---|---|
| `tiny_fast` | 8.6M | Quick experiments | 173K |
| `fast_2060` | 18.9M | RTX 2060 (no MTP) | 95K |
| `fast_2060_mtp` | 19.2M | RTX 2060 with MTP | 57K |
| `modern` | 42.2M | Larger GPUs (24GB+) | 48K |
| `recommended` | 77M | Full featured | — |
| `base` | 46.5M | Vanilla GPT baseline | — |

Find the optimal profile for your hardware:

```bash
python tune_training.py
python tune_training.py --profiles fast_2060_mtp modern recommended
```

## Optimizers

| Optimizer | Description |
|---|---|
| **AdamW** | Standard adaptive optimizer (default) |
| **Muon** | Newton-Schulz momentum — faster convergence for 2D+ weight matrices, paired with AdamW for embeddings/norms |

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

# All options
python train.py --help
```

### 4. Monitor training live

Open a second terminal:

```bash
python -c "import trackio; trackio.show(project='train-from-scratch')"
```

Opens a Gradio dashboard at `http://127.0.0.1:7860` with real-time loss curves, learning rate, and throughput graphs.

### 5. Generate text

```bash
# Default prompts + interactive mode
python generate.py

# Custom prompt
python generate.py --prompt "Once upon a time,"

# With speculative decoding (uses MTP heads, ~30x faster)
python generate.py --no-turboquant --speculative

# Adjust sampling
python generate.py --temperature 0.6 --top-k 50 --max-tokens 300

# Greedy decoding
python generate.py --temperature 0

# All options
python generate.py --help
```

### 6. Evaluate checkpoints

```bash
python evaluate.py
```

Runs all checkpoints against the validation set, prints loss and perplexity:

```
Checkpoint                         Val Loss   Perplexity
-------------------------------------------------------
step_1000.pt                         2.8432        17.17
step_2000.pt                         2.1856         8.90
final.pt                             1.8731         6.51
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

On RTX 2060 Super (8GB VRAM):

| Profile | Steps | Final Loss | Val Loss | Time | tok/s |
|---|---|---|---|---|---|
| `fast_2060_mtp` (batch 32) | 3,000 | 1.87 | ~1.90 | ~7 min | 56K |
| `fast_2060` (batch 40) | 3,000 | 1.87 | ~1.90 | ~18 min | 90K |

## Project Structure

```
config.py              Shared device, dtype, and path configuration
model.py               GPT model with all 8 configurable techniques
train.py               Training loop with CLI, profiles, Muon/AdamW, Trackio
train_tokenizer.py     BPE tokenizer training (16K vocab)
download_data.py       TinyStories dataset download from HuggingFace
generate.py            Text generation with speculative decoding + interactive mode
evaluate.py            Validation loss and perplexity across checkpoints
benchmark.py           Speed and memory benchmarks across all configs
test_techniques.py     Correctness tests for all techniques
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
