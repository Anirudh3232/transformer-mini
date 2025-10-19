FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install black isort flake8

COPY . .
RUN pytest -q

CMD ["python", "scripts/infer.py", "--text", "Hello_2025"]