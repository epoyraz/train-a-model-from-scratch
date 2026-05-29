import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tokenizers import Tokenizer
from model import GPT
from config import DEVICE, DTYPE, DATA_DIR, TOKENIZER_PATH, CHECKPOINT_DIR, VAL_TOKENS_CACHE

BATCH_SIZE = 64


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


def tokenize_validation(block_size):
    if os.path.exists(VAL_TOKENS_CACHE):
        return np.load(VAL_TOKENS_CACHE)

    print("Tokenizing validation set...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ds = load_from_disk(DATA_DIR)
    eos_id = tokenizer.token_to_id("<|eos|>")

    all_ids = []
    for e in tokenizer.encode_batch(ds["validation"]["text"]):
        all_ids.extend(e.ids)
        all_ids.append(eos_id)

    all_ids = np.array(all_ids, dtype=np.uint16)
    all_ids = all_ids[:((len(all_ids) - 1) // block_size) * block_size + 1]
    np.save(VAL_TOKENS_CACHE, all_ids)
    return all_ids


@torch.no_grad()
def eval_checkpoint(path, loader):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE, dtype=DTYPE, enabled=(DEVICE == "cuda")):
            _, loss = model(x, y)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    checkpoints = sorted(glob.glob(f"{CHECKPOINT_DIR}/step_*.pt")) + glob.glob(f"{CHECKPOINT_DIR}/final.pt")
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}/")
        return

    first_ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    block_size = first_ckpt["config"]["block_size"]
    tokens = tokenize_validation(block_size)
    dataset = TokenDataset(tokens, block_size)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Validation: {len(tokens):,} tokens, {len(dataset)} chunks, block_size={block_size}\n")

    print(f"{'Checkpoint':<30} {'Val Loss':>10} {'Perplexity':>12}")
    print("-" * 55)
    for path in checkpoints:
        loss = eval_checkpoint(path, loader)
        ppl = np.exp(loss)
        name = os.path.basename(path)
        print(f"{name:<30} {loss:>10.4f} {ppl:>12.2f}")


if __name__ == "__main__":
    main()
