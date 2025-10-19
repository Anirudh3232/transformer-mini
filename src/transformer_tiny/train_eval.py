from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ModelConfig, TrainConfig
from .model import Seq2SeqTransformer
from .tokenizer import PAD_ID
from .utils import save_checkpoint, set_seed

console = Console()


def build_model(cfg: ModelConfig) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device, pad_id: int) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        # teacher forcing: shift target by one
        logits = model(x, y[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, pad_id: int) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="valid", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x, y[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_loop(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, tcfg: TrainConfig
):
    device = torch.device("cuda" if torch.cuda.is_available() and tcfg.device == "cuda" else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    best = 1e9
    for epoch in range(1, tcfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, PAD_ID)
        val_loss = evaluate(model, val_loader, device, PAD_ID)
        console.print(f"[bold]Epoch {epoch}[/] tr_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        save_checkpoint(f"{tcfg.ckpt_dir}/last.pt", {"model": model.state_dict()})
        if val_loss < best:
            best = val_loss
            save_checkpoint(f"{tcfg.ckpt_dir}/best.pt", {"model": model.state_dict()})
            console.print(f":trophy: Saved best (val_loss={val_loss:.4f})")
