from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import ByteTokenizer, PAD_ID



ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:.,!?@#$%^&*()"


def random_string(min_len: int = 5, max_len: int = 30) -> str:
    L = random.randint(min_len, max_len)
    return "".join(random.choice(ALPHABET) for _ in range(L))


class ReverseTextDataset(Dataset):
    def __init__(self, size: int, max_len: int, tokenizer: ByteTokenizer):
        self.size = size
        self.max_len = max_len
        self.tok = tokenizer
        self.samples = [random_string(5, max_len - 4) for _ in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        s = self.samples[idx]
        inp_ids = self.tok.encode(s, max_len=self.max_len)
        tgt_ids = self.tok.encode(s[::-1], max_len=self.max_len)
        return inp_ids, tgt_ids


def pad_batch(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_fn(batch):
    inputs, targets = zip(*batch)
    x = pad_batch(list(inputs), PAD_ID)
    y = pad_batch(list(targets), PAD_ID)
    return x, y


@dataclass
class DataModule:
    train_ds: ReverseTextDataset
    val_ds: ReverseTextDataset
    tok: ByteTokenizer

    @classmethod
    def build(
        cls, train_size: int, val_size: int, max_len: int, tokenizer: ByteTokenizer
    ) -> "DataModule":
        train = ReverseTextDataset(train_size, max_len, tokenizer)
        val = ReverseTextDataset(val_size, max_len, tokenizer)
        return cls(train_ds=train, val_ds=val, tok=tokenizer)

    def train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def val_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
