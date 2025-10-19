from __future__ import annotations
from dataclasses import dataclass
from typing import List

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259

@dataclass
class ByteTokenizer:
    add_bos: bool = True
    add_eos: bool = True

    def encode(self, text: str, max_len: int | None = None) -> List[int]:
        ids = []
        if self.add_bos:
            ids.append(BOS_ID)
        ids.extend(text.encode("utf-8"))
        if self.add_eos:
            ids.append(EOS_ID)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        filtered = [b for b in ids if b < 256]
        try:
            return bytes(filtered).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE