import torch
from tokenizers import Tokenizer
from model import GPT, MODEL_CONFIG
from config import DEVICE, CHECKPOINT_DIR, TOKENIZER_PATH

CHECKPOINT = f"{CHECKPOINT_DIR}/final.pt"


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = load_model(CHECKPOINT)
    print(f"Loaded {CHECKPOINT} on {DEVICE}\n")

    prompts = [
        "Once upon a time,",
        "The little dog was",
        "One day, a girl named Lily",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print(f"Output: {generate(model, tokenizer, prompt)}")
        print("-" * 60)

    print("\nInteractive mode (type 'quit' to exit):")
    while True:
        prompt = input("\n> ")
        if prompt.lower() == "quit":
            break
        print(generate(model, tokenizer, prompt))


if __name__ == "__main__":
    main()
