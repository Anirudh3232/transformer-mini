.PHONY: format lint test train eval infer

format:
	black src scripts tests
	isort src scripts tests

lint:
	flake8 src scripts tests

test:
	pytest -q

train:
	python scripts/train.py

eval:
	python scripts/eval.py

infer:
	python scripts/infer.py --text "Hello_2025"