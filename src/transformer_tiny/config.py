from dataclasses import dataclass


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 128
    vocab_size: int = 259  # 256 bytes + PAD/BOS/EOS


@dataclass
class DataConfig:
    max_len: int = 64
    train_size: int = 5000
    val_size: int = 500
    batch_size: int = 128


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cuda"  # auto-detect in code if not available
    seed: int = 42
    ckpt_dir: str = "examples/checkpoints"
