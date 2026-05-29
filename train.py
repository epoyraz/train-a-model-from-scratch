import argparse
import math
import os
import shutil
import time

import numpy as np
import torch
import trackio
from datasets import load_from_disk
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, DTYPE, TOKENIZER_PATH, TRAIN_TOKENS_CACHE, VAL_TOKENS_CACHE
from model import CONFIGS, GPT, get_model_config


DEFAULT_BATCH_SIZE = 40
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_MAX_STEPS = 3000
DEFAULT_WARMUP_STEPS = 200
DEFAULT_ADAMW_LR = 6e-4
DEFAULT_MUON_LR = 3e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SAVE_EVERY = 1000
DEFAULT_LOG_EVERY = 10
DEFAULT_EVAL_EVERY = 250
DEFAULT_EVAL_BATCHES = 50


def _truthy(config, key):
    return bool(config.get(key, False))


def _quantization_tag(config):
    for key in ("weight_quant_bits", "quantization_bits", "quant_bits"):
        bits = config.get(key)
        if bits:
            return f"Q{int(bits)}"
    return None


def build_model_name(config, dataset_name="tinystories"):
    n_params = _estimate_params(config)
    tags = []
    if _truthy(config, "use_mtp"):
        tags.append("mtp")
    if _truthy(config, "use_mhc"):
        tags.append("mhc")
    if _truthy(config, "use_fast_bitnet"):
        tags.append("fbitnet")
    elif _truthy(config, "use_bitnet"):
        tags.append("bitnet")
    suffix = "-".join(tags)
    if suffix:
        return f"{dataset_name}-{n_params}-{suffix}"
    return f"{dataset_name}-{n_params}"


def _estimate_params(config):
    n = config["vocab_size"] * config["n_embd"]
    n += config["n_layer"] * (4 * config["n_embd"] ** 2 + 3 * config["n_embd"] * (4 * config["n_embd"]))
    if n >= 1e9:
        return f"{n/1e9:.0f}b"
    return f"{n/1e6:.0f}m"


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
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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
                buf_2d = buf.view(shape[0], -1) if buf.ndim > 2 else buf
                update = newton_schulz(buf_2d, steps=group["ns_steps"])
                if buf.ndim > 2:
                    update = update.view(shape)
                scale = max(1, buf_2d.shape[0] / buf_2d.shape[1]) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small GPT language model.")
    parser.add_argument("--profile", default=os.getenv("TRAIN_PROFILE", "fast_2060"), choices=sorted(CONFIGS))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)))
    parser.add_argument("--grad-accum-steps", type=int, default=int(os.getenv("GRAD_ACCUM_STEPS", DEFAULT_GRAD_ACCUM_STEPS)))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", DEFAULT_MAX_STEPS)))
    parser.add_argument("--warmup-steps", type=int, default=int(os.getenv("WARMUP_STEPS", DEFAULT_WARMUP_STEPS)))
    parser.add_argument("--max-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY)))
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default=os.getenv("OPTIMIZER", "adamw"))
    parser.add_argument("--save-every", type=int, default=int(os.getenv("SAVE_EVERY", DEFAULT_SAVE_EVERY)))
    parser.add_argument("--log-every", type=int, default=int(os.getenv("LOG_EVERY", DEFAULT_LOG_EVERY)))
    parser.add_argument("--eval-every", type=int, default=int(os.getenv("EVAL_EVERY", DEFAULT_EVAL_EVERY)))
    parser.add_argument("--eval-batches", type=int, default=int(os.getenv("EVAL_BATCHES", DEFAULT_EVAL_BATCHES)))
    parser.add_argument("--checkpoint-dir", default=os.getenv("CHECKPOINT_DIR", CHECKPOINT_DIR))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", 2)))
    parser.add_argument("--compile", action="store_true", default=os.getenv("TORCH_COMPILE", "0") == "1")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--no-trackio", action="store_true")
    return parser.parse_args()


def tokenize_split(split_name, cache_path, block_size):
    if os.path.exists(cache_path):
        print(f"Loading cached {split_name} tokens...")
        return np.load(cache_path)

    print(f"Tokenizing {split_name} split...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ds = load_from_disk(DATA_DIR)
    if split_name not in ds:
        raise ValueError(f"Dataset at {DATA_DIR!r} has no {split_name!r} split")
    eos_id = tokenizer.token_to_id("<|eos|>")

    all_ids = []
    split = ds[split_name]
    for i in range(0, len(split), 1000):
        batch = split[i : i + 1000]["text"]
        encoded = tokenizer.encode_batch(batch)
        for e in encoded:
            all_ids.extend(e.ids)
            all_ids.append(eos_id)
        if i % 100000 == 0:
            print(f"  {i:,} / {len(split):,}")

    all_ids = np.array(all_ids, dtype=np.uint16)
    all_ids = all_ids[: ((len(all_ids) - 1) // block_size) * block_size + 1]
    np.save(cache_path, all_ids)
    print(f"Tokenized {split_name}: {len(all_ids):,} tokens ({cache_path}: {os.path.getsize(cache_path) / 1e6:.0f} MB)")
    return all_ids


def tokenize_dataset(block_size):
    return tokenize_split("train", TRAIN_TOKENS_CACHE, block_size)


def tokenize_validation(block_size):
    return tokenize_split("validation", VAL_TOKENS_CACHE, block_size)


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizers(model, optimizer_name, max_lr, weight_decay):
    fused = DEVICE == "cuda"
    if optimizer_name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay, fused=fused)]

    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "emb" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizers = [Muon(muon_params, lr=max_lr, momentum=0.95)]
    if adamw_params:
        optimizers.append(torch.optim.AdamW(adamw_params, lr=max_lr / 3, weight_decay=weight_decay, fused=fused))
    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adamw_params):,}")
    return optimizers


def set_optimizer_lr(optimizers, lr, optimizer_name):
    for idx, opt in enumerate(optimizers):
        group_lr = lr
        if optimizer_name == "muon" and idx > 0:
            group_lr = lr / 3
        for pg in opt.param_groups:
            pg["lr"] = group_lr


def save_checkpoint(path, model, step, config, args, model_name):
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "config": config,
            "train_args": vars(args),
            "model_name": model_name,
        },
        path,
    )


def mirror_checkpoint(src_path, dst_path):
    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        return
    if os.path.exists(dst_path):
        os.remove(dst_path)
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


def save_named_checkpoint(checkpoint_dir, model_name, model, step, config, args, final=False):
    suffix = "final" if final else f"step_{step}"
    named_path = os.path.join(checkpoint_dir, f"{model_name}_{suffix}.pt")
    latest_path = os.path.join(checkpoint_dir, f"{suffix}.pt")
    save_checkpoint(named_path, model, step, config, args, model_name)
    mirror_checkpoint(named_path, latest_path)
    return named_path, latest_path


@torch.no_grad()
def evaluate_loss(model, loader, max_batches):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches = 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
        total_loss += loss.item()
        batches += 1
        if batches >= max_batches:
            break
    if was_training:
        model.train()
    return total_loss / max(1, batches)


def main():
    args = parse_args()
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True

    max_lr = args.max_lr
    if max_lr is None:
        max_lr = DEFAULT_MUON_LR if args.optimizer == "muon" else DEFAULT_ADAMW_LR
    min_lr = args.min_lr if args.min_lr is not None else max_lr * 0.1

    model_config = get_model_config(args.profile)
    if args.activation_checkpointing:
        model_config["use_activation_checkpointing"] = True
    block_size = model_config["block_size"]
    model_name = build_model_name(model_config)

    torch.manual_seed(42)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not args.no_trackio:
        trackio.init(
            project="train-from-scratch",
            name=model_name,
            config={
                "model_name": model_name,
                "profile": args.profile,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "max_steps": args.max_steps,
                "max_lr": max_lr,
                "min_lr": min_lr,
                "optimizer": args.optimizer,
                "compile": args.compile,
                "eval_every": args.eval_every,
                "eval_batches": args.eval_batches,
                "checkpoint_dir": args.checkpoint_dir,
                **model_config,
            },
        )

    tokens = tokenize_dataset(block_size)
    dataset = TokenDataset(tokens, block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = None
    val_tokens = None
    if args.eval_every > 0 and args.eval_batches > 0:
        val_tokens = tokenize_validation(block_size)
        val_dataset = TokenDataset(val_tokens, block_size)
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(DEVICE == "cuda"),
                persistent_workers=(args.num_workers > 0),
            )
        else:
            print("Validation split is too small for this block size; skipping eval.")

    raw_model = GPT(model_config).to(DEVICE)
    model = raw_model
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    n_params = sum(p.numel() for p in raw_model.parameters())
    optimizers = build_optimizers(model, args.optimizer, max_lr, args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Profile: {args.profile}")
    print(f"Checkpoint name: {model_name}")
    print(f"Model: {n_params:,} params")
    print(f"Dataset: {len(dataset):,} chunks, {len(tokens):,} tokens")
    if val_loader is not None:
        print(f"Validation: {len(val_loader.dataset):,} chunks, {len(val_tokens):,} tokens, every {args.eval_every} steps")
    print(f"Micro batch: {args.batch_size} x {block_size} = {args.batch_size * block_size:,} tokens")
    print(f"Effective batch: {args.batch_size * args.grad_accum_steps} sequences = {args.batch_size * args.grad_accum_steps * block_size:,} tokens/step")
    print(f"Optimizer: {args.optimizer}, lr {max_lr:.2e} -> {min_lr:.2e}")
    print(f"Training for {args.max_steps} steps...\n")

    step = 0
    running_loss = 0.0
    t0 = time.time()
    data_iter = iter(loader)

    while step < args.max_steps:
        lr = get_lr(step, max_lr, min_lr, args.warmup_steps, args.max_steps)
        set_optimizer_lr(optimizers, lr, args.optimizer)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        accum_loss = 0.0
        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.detach().item()

        for opt in optimizers:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()

        running_loss += accum_loss
        step += 1

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t0
            tokens_per_sec = args.log_every * args.batch_size * args.grad_accum_steps * block_size / elapsed
            print(f"step {step:>6} | loss {avg_loss:.4f} | lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s")
            if not args.no_trackio:
                trackio.log({"loss": avg_loss, "lr": lr, "tokens_per_sec": tokens_per_sec})
            running_loss = 0.0
            t0 = time.time()

        if val_loader is not None and step % args.eval_every == 0:
            val_loss = evaluate_loss(model, val_loader, args.eval_batches)
            print(f"           | val_loss {val_loss:.4f}")
            if not args.no_trackio:
                trackio.log({"val_loss": val_loss})
            t0 = time.time()

        if step % args.save_every == 0:
            named_path, latest_path = save_named_checkpoint(
                args.checkpoint_dir,
                model_name,
                raw_model,
                step,
                model_config,
                args,
            )
            print(f"  Saved {named_path}")
            print(f"  Updated {latest_path}")

    named_path, latest_path = save_named_checkpoint(
        args.checkpoint_dir,
        model_name,
        raw_model,
        step,
        model_config,
        args,
        final=True,
    )
    print(f"\nTraining done. Saved {named_path}")
    print(f"Latest checkpoint alias: {latest_path}")


if __name__ == "__main__":
    main()
