from datasets import load_dataset
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading TinyStories from HuggingFace...")
ds = load_dataset("roneneldan/TinyStories")

print(f"Train: {len(ds['train']):,} examples")
print(f"Validation: {len(ds['validation']):,} examples")

ds.save_to_disk(DATA_DIR)
print(f"Saved to {DATA_DIR}")
