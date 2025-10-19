from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import positional_encoding, causal_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # x_q: [B, Lq, D], x_kv: [B, Lk, D]
        B, Lq, _ = x_q.shape
        Lk = x_kv.size(1)

        q = self.q_proj(x_q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lq, Dh]
        k = self.k_proj(x_kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lk, Dh]
        v = self.v_proj(x_kv).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, Lk, Dh]

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, Lq, Lk]

        if attn_mask is not None:
            # attn_mask: [Lq, Lk] or [B, H, Lq, Lk] (we use [L,L])
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask: [B, Lk] -> True for pads; expand to [B, 1, 1, Lk]
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # [B, H, Lq, Dh]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.o_proj(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        # Self-attention
        sa = self.self_attn(x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.drop(sa))
        # FFN
        ff = self.ffn(x)
        x = self.norm2(x + self.drop(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        # masked self-attention
        sa = self.self_attn(x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.drop(sa))
        # cross attention
        ca = self.cross_attn(x, enc_out, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.drop(ca))
        # FFN
        ff = self.ffn(x)
        x = self.norm3(x + self.drop(ff))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 dropout: float, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.pe = None  # built lazily
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _get_pe(self, device: torch.device):
        if self.pe is None or self.pe.device != device or self.pe.size(0) < self.max_len:
            self.pe = positional_encoding(self.max_len, self.d_model, device)  # [L, D]
        return self.pe

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # src/tgt: [B, L]
        device = src.device
        pe = self._get_pe(device)

        src_emb = self.src_emb(src) + pe[: src.size(1)]
        tgt_emb = self.tgt_emb(tgt) + pe[: tgt.size(1)]

        x = src_emb
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        memory = x

        y = tgt_emb
        tgt_mask = causal_mask(tgt.size(1), device)  # [L, L]
        for layer in self.decoder:
            y = layer(y, memory, tgt_mask=tgt_mask,
                      tgt_key_padding_mask=tgt_key_padding_mask,
                      memory_key_padding_mask=src_key_padding_mask)

        logits = self.out_proj(y)  # [B, Lt, V]
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int, pad_id: int,
                      max_new_tokens: int = 64) -> torch.Tensor:
        # src: [B, Ls]
        device = src.device
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        src_pad = (src == pad_id)
        for _ in range(max_new_tokens):
            tgt_pad = (ys == pad_id)
            logits = self.forward(src, ys, src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys