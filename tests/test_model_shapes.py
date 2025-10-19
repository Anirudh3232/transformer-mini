import torch

from transformer_tiny.model import Seq2SeqTransformer
from transformer_tiny.tokenizer import PAD_ID


def test_forward_shapes():
    V = 259
    model = Seq2SeqTransformer(
        vocab_size=V, d_model=64, n_heads=4, n_layers=1, d_ff=128, dropout=0.0, max_len=32
    )
    B, Ls, Lt = 2, 10, 12
    src = torch.randint(0, V, (B, Ls))
    tgt = torch.randint(0, V, (B, Lt))
    logits = model(src, tgt)
    assert logits.shape == (B, Lt, V)
