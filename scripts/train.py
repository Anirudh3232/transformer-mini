from __future__ import annotations

import argparse

from transformer_tiny.config import DataConfig, ModelConfig, TrainConfig
from transformer_tiny.data import ByteTokenizer, DataModule
from transformer_tiny.train_eval import build_model, train_loop
from transformer_tiny.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a tiny seq2seq Transformer (reverse task).")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-size", type=int, default=5000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tok = ByteTokenizer()
    data = DataModule.build(
        train_size=args.train_size, val_size=args.val_size, max_len=args.max_len, tokenizer=tok
    )

    mcfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        vocab_size=tok.vocab_size,
    )
    tcfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed, device=args.device)
    model = build_model(mcfg)

    train_loader = data.train_loader(batch_size=args.batch_size)
    val_loader = data.val_loader(batch_size=args.batch_size)
    train_loop(model, train_loader, val_loader, tcfg)


if __name__ == "__main__":
    main()
