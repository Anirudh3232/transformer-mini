# mini-transformer 🧠⚡

A from-scratch **seq2seq Transformer** (PyTorch) that learns a simple but demonstrative task:
**reverse a string**. It’s small, readable, and production-ish—perfect for showcasing code quality
to recruiters while being quick to run locally or in a Colab.

## Why this repo might impress
- ✨ Clean, typed, human-friendly code (no magic “black box”).
- 🧩 From-scratch attention, encoder/decoder blocks, positional encodings.
- 🧪 Tests (pytest), linting, CI (GitHub Actions), Dockerfile, and Makefile.
- 🛠️ Simple CLI to train/eval/infer. No massive datasets required.
- 📦 Reusable tokenizer and datamodule (byte-level with PAD/BOS/EOS).

> Task: Given a string like `hello123`, predict `321olleh`

---

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Train (CPU okay)
```bash
python scripts/train.py --epochs 3 --batch-size 128
```

### 3) Evaluate
```bash
python scripts/eval.py
```

### 4) Inference
```bash
python scripts/infer.py --text "ChatGPT-5 is awesome!" --max-new-tokens 64
# -> prints the reversed string
```

### 5) Colab
Open `notebooks/colab_quickstart.ipynb` (minimal).

---

## Repo Layout

```
.
├── src/transformer_tiny
│   ├── __init__.py
│   ├── config.py          # dataclasses for hyperparams
│   ├── tokenizer.py       # byte-level tokenizer (PAD/BOS/EOS)
│   ├── data.py            # dataset + dataloaders (reverse task)
│   ├── model.py           # positional enc, attention, encoder/decoder
│   ├── utils.py           # seed, checkpoints, metrics, masks
│   └── train_eval.py      # train/eval steps
├── scripts
│   ├── train.py           # CLI training
│   ├── eval.py            # CLI evaluation
│   └── infer.py           # CLI greedy decoding
├── tests                  # small unit tests
│   ├── test_tokenizer.py
│   └── test_model_shapes.py
├── notebooks
│   └── colab_quickstart.ipynb
├── .github/workflows/ci.yml
├── requirements.txt
├── pyproject.toml         # black/isort/flake8 settings
├── Dockerfile
├── Makefile
├── LICENSE
└── README.md
```

---

## Example Results

After ~3 epochs on CPU, the model should reliably learn to reverse short strings.

```
input : "Hello_2025"
target: "5202_olleH"
pred  : "5202_olleH"
```

---

## Notes

- This is a **teaching/portfolio** repo: readable code > maximal speed.
- Easily adapt `data.py` to new toy tasks (e.g., copy, shift, parentheses balancing).
- For real tasks, hook in a proper dataset/tokenizer and scale the config.

**MIT License** — Have fun!

— Built 2025-10-19