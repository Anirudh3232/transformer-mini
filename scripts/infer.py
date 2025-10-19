from __future__ import annotations

import argparse

import torch

from transformer_tiny.config import ModelConfig
from transformer_tiny.data import ByteTokenizer, DataModule
from transformer_tiny.tokenizer import BOS_ID, EOS_ID, PAD_ID
from transformer_tiny.train_eval import build_model
from transformer_tiny.utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Greedy decode with tiny transformer.")
    p.add_argument("--text", type=str, default="Hello_2025")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ckpt", type=str, default="examples/checkpoints/best.pt")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    tok = ByteTokenizer()
    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        vocab_size=tok.vocab_size,
    )
    model = build_model(cfg)

    # try load checkpoint; if not available, it's fine (random weights demo)
    try:
        ckpt = load_checkpoint(args.ckpt)
        model.load_state_dict(ckpt["model"], strict=False)
    except FileNotFoundError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model.to(device)

    # prepare source
    ids = tok.encode(args.text, max_len=args.max_len)
    src = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.greedy_decode(
        src, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_new_tokens=args.max_new_tokens
    )
    # drop BOS
    out_ids = out[0].tolist()[1:]
    text = tok.decode(out_ids)
    print(text)


if __name__ == "__main__":
    main()
