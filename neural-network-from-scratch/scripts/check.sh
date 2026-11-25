#!/usr/bin/env bash
# "CI local": o mesmo que o GitHub Actions faria. Rode antes de cada commit.
# Uso: bash scripts/check.sh            (tudo)
#      bash scripts/check.sh numpy      (só os testes da Fase 1)
set -e
echo "== Lint =="
ruff check .
echo "== Testes =="
if [ "$1" = "numpy" ]; then
  pytest tests/test_data.py tests/test_numpy_layers.py tests/test_numpy_training.py
else
  pytest --cov=src --cov-report=term-missing
fi
echo "Tudo certo."
