import argparse
import time

import torch
from tokenizers import Tokenizer

from config import CHECKPOINT_DIR, DEVICE, TOKENIZER_PATH
from model import GPT
from msvc_env import ensure_msvc_env


DEFAULT_CHECKPOINT = f"{CHECKPOINT_DIR}/final.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--speculate-tokens", type=int, default=None)
    parser.add_argument("--turboquant", action="store_true")
    parser.add_argument("--no-turboquant", action="store_true")
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the decode step (~2x faster after warmup; needs MSVC+Triton)")
    return parser.parse_args()


def load_model(checkpoint_path):
    probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if probe.get("format") == "ternary_packed":
        from export_ternary import load_ternary
        model, config = load_ternary(checkpoint_path, device=DEVICE)
        return model, {"config": config, "step": probe.get("step", 0)}
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = GPT(ckpt["config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def resolve_turboquant(args, config):
    if args.turboquant:
        return True
    if args.no_turboquant:
        return False
    return config.get("use_turboquant", False)


def generate_text(model, tokenizer, prompt, args, use_turboquant):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            speculative=args.speculative,
            speculate_tokens=args.speculate_tokens,
            use_turboquant=use_turboquant,
            use_kv_cache=not args.no_kv_cache,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated_tokens = out.shape[1] - idx.shape[1]
    tps = generated_tokens / elapsed if elapsed > 0 else float("inf")
    return tokenizer.decode(out[0].tolist()), generated_tokens, elapsed, tps


def print_generation(model, tokenizer, prompt, args, use_turboquant):
    text, generated_tokens, elapsed, tps = generate_text(model, tokenizer, prompt, args, use_turboquant)
    print(f"Prompt: {prompt}")
    print(f"Output: {text}")
    print(f"Generated: {generated_tokens} tokens in {elapsed:.3f}s ({tps:.2f} tok/s)")
    print("-" * 60)


def main():
    args = parse_args()
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model, ckpt = load_model(args.checkpoint)
    config = ckpt["config"]
    use_turboquant = resolve_turboquant(args, config)

    if args.compile:
        if DEVICE == "cuda":
            ensure_msvc_env()  # Triton needs MSVC on PATH to build CUDA shims on Windows
        # Compiling the per-step forward removes Python/launch overhead during decode.
        # The first few generations are slow (compilation + shape specialization).
        model._forward_inference = torch.compile(model._forward_inference)
        print("Compiled decode step (first generation will be slow while compiling).")

    if args.speculative and not config.get("use_mtp", False):
        print("Speculative mode requested, but this checkpoint has no MTP heads. Falling back to normal generation.")
    print(f"Loaded {args.checkpoint} on {DEVICE}")
    print(f"Config: {config}")
    print(f"Mode: speculative={args.speculative and config.get('use_mtp', False)}, kv_cache={not args.no_kv_cache}, turboquant={use_turboquant}\n")

    if args.prompt is not None:
        print_generation(model, tokenizer, args.prompt, args, use_turboquant)
        return

    prompts = [
        "Once upon a time,",
        "The little dog was",
        "One day, a girl named Lily",
    ]
    for prompt in prompts:
        print_generation(model, tokenizer, prompt, args, use_turboquant)

    print("\nInteractive mode (type 'quit' to exit):")
    while True:
        prompt = input("\n> ")
        if prompt.lower() == "quit":
            break
        print_generation(model, tokenizer, prompt, args, use_turboquant)


if __name__ == "__main__":
    main()
