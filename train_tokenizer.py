from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DATA_DIR = "data"
VOCAB_SIZE = 16384
TOKENIZER_PATH = "tokenizer.json"

print("Loading dataset...")
ds = load_from_disk(DATA_DIR)

def batch_iterator(batch_size=1000):
    for i in range(0, len(ds["train"]), batch_size):
        yield ds["train"][i : i + batch_size]["text"]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|pad|>", "<|eos|>", "<|unk|>"],
    show_progress=True,
)

print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.save(TOKENIZER_PATH)

test = "Once upon a time, a little girl named Lily went to the park."
encoded = tokenizer.encode(test)
print(f"\nVocab size: {tokenizer.get_vocab_size()}")
print(f"Test: {test}")
print(f"Tokens: {encoded.tokens[:20]}")
print(f"IDs: {encoded.ids[:20]}")
print(f"Decoded: {tokenizer.decode(encoded.ids)}")
print(f"\nSaved to {TOKENIZER_PATH}")
