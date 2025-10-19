from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, state: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, d_model, dtype=torch.float32, device=device).unsqueeze(0)
    angle_rates = 1.0 / (10000 ** (i / d_model))
    angles = pos * angle_rates
    pe = torch.zeros((max_len, d_model), device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe  # [max_len, d_model]


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # [L, L] mask for decoder self-attn: True above diagonal (masked)
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


def padding_mask(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
    # seq: [B, L] -> True where PAD
    return seq == pad_id


def exact_match(pred: str, tgt: str) -> bool:
    return pred == tgt
