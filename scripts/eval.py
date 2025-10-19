from __future__ import annotations
import argparse
import torch

from transformer_tiny.config import ModelConfig, DataConfig
from transformer_tiny.data import DataModule, ByteTokenizer
from transformer_tiny.train_eval import build_model, evaluate
from transformer_tiny.utils import load_checkpoint
from transformer_tiny.tokenizer import PAD_ID

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tiny transformer on reverse task.")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ckpt", type=str, default="examples/checkpoints/best.pt")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()

def main():
    args = parse_args()
    tok = ByteTokenizer()
    data = DataModule.build(train_size=100, val_size=args.val_size, max_len=args.max_len, tokenizer=tok)

    cfg = ModelConfig(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                      d_ff=args.d_ff, dropout=args.dropout, max_len=args.max_len, vocab_size=tok.vocab_size)
    model = build_model(cfg)

    ckpt = load_checkpoint(args.ckpt)
    model.load_state_dict(ckpt["model"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model.to(device)

    val_loader = data.val_loader(batch_size=args.batch_size)
    val_loss = evaluate(model, val_loader, device, PAD_ID)
    print(f"val_loss={val_loss:.4f}")

if __name__ == "__main__":
    main()