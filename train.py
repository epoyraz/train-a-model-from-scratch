import os
import time
import math
import numpy as np
import torch
import trackio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tokenizers import Tokenizer
from model import GPT, MODEL_CONFIG
from config import DEVICE, DTYPE, DATA_DIR, TOKENIZER_PATH, CHECKPOINT_DIR, TRAIN_TOKENS_CACHE

# --- Config ---
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
BLOCK_SIZE = MODEL_CONFIG["block_size"]
MAX_STEPS = 3000
WARMUP_STEPS = 200
MAX_LR = 3e-3
MIN_LR = 3e-4
EMBED_LR = 1e-3
WEIGHT_DECAY = 0.01
SAVE_EVERY = 1000
LOG_EVERY = 10

# --- Muon optimizer ---
@torch.no_grad()
def newton_schulz(M, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M / (M.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-3, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    p.add_(g, alpha=-lr)
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                shape = buf.shape
                if buf.ndim > 2:
                    buf_2d = buf.view(shape[0], -1)
                else:
                    buf_2d = buf
                update = newton_schulz(buf_2d, steps=group["ns_steps"])
                if buf.ndim > 2:
                    update = update.view(shape)
                scale = max(1, buf_2d.shape[0] / buf_2d.shape[1]) ** 0.5
                p.add_(update, alpha=-lr * scale)


# --- Dataset ---
class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = torch.from_numpy(self.tokens[start : start + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1 : start + self.block_size + 1].astype(np.int64))
        return x, y


def tokenize_dataset():
    cache_path = TRAIN_TOKENS_CACHE
    if os.path.exists(cache_path):
        print("Loading cached tokens...")
        return np.load(cache_path)

    print("Tokenizing dataset...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ds = load_from_disk(DATA_DIR)
    eos_id = tokenizer.token_to_id("<|eos|>")

    all_ids = []
    for i in range(0, len(ds["train"]), 1000):
        batch = ds["train"][i : i + 1000]["text"]
        encoded = tokenizer.encode_batch(batch)
        for e in encoded:
            all_ids.extend(e.ids)
            all_ids.append(eos_id)
        if i % 100000 == 0:
            print(f"  {i:,} / {len(ds['train']):,}")

    all_ids = np.array(all_ids, dtype=np.uint16)
    all_ids = all_ids[:((len(all_ids) - 1) // BLOCK_SIZE) * BLOCK_SIZE + 1]
    np.save(cache_path, all_ids)
    print(f"Tokenized: {len(all_ids):,} tokens ({cache_path}: {os.path.getsize(cache_path) / 1e6:.0f} MB)")
    return all_ids


def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def main():
    torch.manual_seed(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    trackio.init(
        project="train-from-scratch",
        name="tinystories-50m",
        config={
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "block_size": BLOCK_SIZE,
            "max_steps": MAX_STEPS,
            "max_lr": MAX_LR,
            "optimizer": "muon+adamw",
            **MODEL_CONFIG,
        },
    )

    tokens = tokenize_dataset()
    dataset = TokenDataset(tokens, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = GPT(MODEL_CONFIG).to(DEVICE)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Split params: Muon for 2D+ weights, AdamW for embeddings/norms/biases
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "emb" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = Muon(muon_params, lr=MAX_LR, momentum=0.95)
    optimizer_embed = torch.optim.AdamW(adamw_params, lr=EMBED_LR, weight_decay=WEIGHT_DECAY, fused=True)
    scaler = torch.amp.GradScaler()

    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")
    print(f"Dataset: {len(dataset):,} chunks, {len(tokens):,} tokens")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS} sequences = {BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE:,} tokens/step")
    print(f"Training for {MAX_STEPS} steps...\n")

    step = 0
    running_loss = 0.0
    t0 = time.time()
    data_iter = iter(loader)

    while step < MAX_STEPS:
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        for pg in optimizer_embed.param_groups:
            pg["lr"] = lr * (EMBED_LR / MAX_LR)

        optimizer.zero_grad(set_to_none=True)
        optimizer_embed.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        scaler.unscale_(optimizer_embed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.step(optimizer_embed)
        scaler.update()

        running_loss += accum_loss
        step += 1

        if step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            elapsed = time.time() - t0
            tokens_per_sec = LOG_EVERY * BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE / elapsed
            print(f"step {step:>6} | loss {avg_loss:.4f} | lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s")
            trackio.log({"loss": avg_loss, "lr": lr, "tokens_per_sec": tokens_per_sec})
            running_loss = 0.0
            t0 = time.time()

        if step % SAVE_EVERY == 0:
            path = f"checkpoints/step_{step}.pt"
            torch.save({"step": step, "model": model.state_dict(), "config": MODEL_CONFIG}, path)
            print(f"  Saved {path}")

    path = "checkpoints/final.pt"
    torch.save({"step": step, "model": model.state_dict(), "config": MODEL_CONFIG}, path)
    print(f"\nTraining done. Saved {path}")


if __name__ == "__main__":
    main()
