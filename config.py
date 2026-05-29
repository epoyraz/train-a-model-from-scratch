import torch

DATA_DIR = "data"
TOKENIZER_PATH = "tokenizer.json"
CHECKPOINT_DIR = "checkpoints"
TRAIN_TOKENS_CACHE = "data/train_tokens.npy"
VAL_TOKENS_CACHE = "data/val_tokens.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
