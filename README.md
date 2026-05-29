# Train a Language Model from Scratch

A from-scratch GPT-style language model trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) with modern architecture techniques.

## Architecture

50M parameter decoder-only transformer with configurable cutting-edge techniques:

| Technique | Description | Config Flag |
|---|---|---|
| **RoPE** | Rotary Position Embeddings (LLaMA-style) | `use_rope: True` |
| **GQA** | Grouped Query Attention — shared KV heads | `n_kv_head: 2` |
| **SwiGLU** | Gated MLP activation (LLaMA, DeepSeek) | `use_swiglu: True` |
| **RMSNorm** | Simpler, faster normalization | `use_rmsnorm: True` |
| **MTP** | Multi-Token Prediction (DeepSeek-V3) | `use_mtp: True` |
| **mHC** | Manifold-Constrained Hyper-Connections (DeepSeek) | `use_mhc: True` |
| **BitNet** | Ternary 1-bit weights with STE | `use_bitnet: True` |
| **TurboQuant** | KV-cache compression for inference (Google) | `use_turboquant: True` |

All techniques are independently toggleable. The recommended config combines RoPE + GQA + SwiGLU + RMSNorm + MTP.

## Training

Uses the **Muon optimizer** (Newton-Schulz momentum) for 2D+ weight matrices and AdamW for embeddings/norms, with cosine LR schedule and fp16 mixed precision.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install datasets tokenizers trackio
```

## Usage

### 1. Download data
```bash
python download_data.py
```

### 2. Train tokenizer
```bash
python train_tokenizer.py
```

### 3. Train model
```bash
python train.py
```
View live training dashboard in a second terminal:
```bash
python -c "import trackio; trackio.show(project='train-from-scratch')"
```

### 4. Generate text
```bash
python generate.py
```

### 5. Evaluate checkpoints
```bash
python evaluate.py
```

### 6. Benchmark configs
```bash
python benchmark.py
```

### 7. Run tests
```bash
python test_techniques.py
```

## Project Structure

```
config.py            # Shared device/path configuration
model.py             # GPT model with all configurable techniques
train.py             # Training loop with Muon optimizer + Trackio logging
train_tokenizer.py   # BPE tokenizer training
download_data.py     # TinyStories dataset download
generate.py          # Text generation with interactive mode
evaluate.py          # Validation loss + perplexity across checkpoints
benchmark.py         # Speed/memory benchmarks across configs
test_techniques.py   # Correctness tests for all techniques
```

## Hardware

Developed and tested on an NVIDIA RTX 2060 Super (8GB VRAM). Training takes ~5 hours at batch size 8 with gradient accumulation.
