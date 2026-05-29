import argparse
import os
import torch
from model import GPT, FastBitLinear
from config import DEVICE, CHECKPOINT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Export FastBitLinear model to packed ternary format.")
    parser.add_argument("--checkpoint", default=f"{CHECKPOINT_DIR}/final.pt")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def pack_ternary(w_pos, w_neg):
    """Pack two binary masks into 2 bits per weight using uint8.
    Each byte holds 4 ternary weights: 2 bits each (pos_bit, neg_bit).
    00 = zero, 10 = positive, 01 = negative.
    """
    assert w_pos.shape == w_neg.shape
    flat_pos = w_pos.flatten().to(torch.uint8)
    flat_neg = w_neg.flatten().to(torch.uint8)
    n = flat_pos.shape[0]
    pad = (4 - n % 4) % 4
    if pad:
        flat_pos = torch.cat([flat_pos, torch.zeros(pad, dtype=torch.uint8)])
        flat_neg = torch.cat([flat_neg, torch.zeros(pad, dtype=torch.uint8)])
    flat_pos = flat_pos.view(-1, 4)
    flat_neg = flat_neg.view(-1, 4)
    packed = (
        (flat_pos[:, 0] << 7) | (flat_neg[:, 0] << 6) |
        (flat_pos[:, 1] << 5) | (flat_neg[:, 1] << 4) |
        (flat_pos[:, 2] << 3) | (flat_neg[:, 2] << 2) |
        (flat_pos[:, 3] << 1) | (flat_neg[:, 3] << 0)
    )
    return packed.to(torch.uint8), n


def unpack_ternary(packed, n, shape):
    """Unpack packed ternary back to pos/neg masks."""
    pos_bits = []
    neg_bits = []
    for shift in [7, 5, 3, 1]:
        pos_bits.append((packed >> shift) & 1)
    for shift in [6, 4, 2, 0]:
        neg_bits.append((packed >> shift) & 1)
    flat_pos = torch.stack(pos_bits, dim=1).flatten()[:n]
    flat_neg = torch.stack(neg_bits, dim=1).flatten()[:n]
    return flat_pos.view(shape).to(torch.int8), flat_neg.view(shape).to(torch.int8)


def export_model(checkpoint_path, output_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    exported = {
        "config": config,
        "step": ckpt.get("step", 0),
        "format": "ternary_packed",
    }

    n_ternary = 0
    n_regular = 0
    ternary_bytes = 0
    regular_bytes = 0

    for name, module in model.named_modules():
        if isinstance(module, FastBitLinear):
            w = module.weight.data
            alpha = w.abs().mean()
            threshold = alpha * 0.5
            w_pos = (w > threshold).to(torch.int8)
            w_neg = (w < -threshold).to(torch.int8)
            packed, n_elements = pack_ternary(w_pos, w_neg)

            exported[f"{name}.packed"] = packed
            exported[f"{name}.alpha"] = alpha
            exported[f"{name}.shape"] = list(w.shape)
            exported[f"{name}.n_elements"] = n_elements
            if module.bias is not None:
                exported[f"{name}.bias"] = module.bias.data

            n_ternary += 1
            ternary_bytes += packed.numel()
            continue

    for name, param in model.named_parameters():
        key_base = name.rsplit(".", 1)[0] if "." in name else ""
        is_ternary = any(f"{key_base}.packed" in k for k in exported)
        if not is_ternary and f"{name}" not in exported:
            exported[name] = param.data
            n_regular += 1
            regular_bytes += param.numel() * param.element_size()

    torch.save(exported, output_path)

    file_size = os.path.getsize(output_path)
    orig_size = os.path.getsize(checkpoint_path)

    print(f"Exported to {output_path}")
    print(f"  Ternary layers: {n_ternary}")
    print(f"  Regular params: {n_regular}")
    print(f"  Original size:  {orig_size / 1e6:.1f} MB")
    print(f"  Exported size:  {file_size / 1e6:.1f} MB")
    print(f"  Compression:    {orig_size / file_size:.1f}x")

    verify_roundtrip(model, exported)


def verify_roundtrip(model, exported):
    print("\nVerifying roundtrip...")
    errors = 0
    for name, module in model.named_modules():
        if isinstance(module, FastBitLinear):
            w = module.weight.data
            alpha_orig = w.abs().mean()
            threshold = alpha_orig * 0.5
            w_pos_orig = (w > threshold).to(torch.int8)
            w_neg_orig = (w < -threshold).to(torch.int8)

            packed = exported[f"{name}.packed"]
            n_elements = exported[f"{name}.n_elements"]
            shape = exported[f"{name}.shape"]
            alpha_loaded = exported[f"{name}.alpha"]

            w_pos_rt, w_neg_rt = unpack_ternary(packed, n_elements, shape)

            if not torch.equal(w_pos_orig, w_pos_rt):
                print(f"  FAIL  {name} pos mask mismatch")
                errors += 1
            elif not torch.equal(w_neg_orig, w_neg_rt):
                print(f"  FAIL  {name} neg mask mismatch")
                errors += 1
            elif abs(alpha_orig.item() - alpha_loaded.item()) > 1e-6:
                print(f"  FAIL  {name} alpha mismatch")
                errors += 1

    if errors == 0:
        print("  All ternary layers verified - perfect roundtrip!")
    else:
        print(f"  {errors} layer(s) failed roundtrip!")


def main():
    args = parse_args()
    output = args.output
    if output is None:
        base = os.path.splitext(args.checkpoint)[0]
        output = f"{base}_ternary.pt"
    export_model(args.checkpoint, output)


if __name__ == "__main__":
    main()
