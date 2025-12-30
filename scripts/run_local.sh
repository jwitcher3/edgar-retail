#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"

# Always run with the venv python to avoid conda/path issues
PY=".venv/bin/python"

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG"
  echo "Create it with: cp config.example.yaml config.yaml"
  exit 1
fi

echo "==> Ingest"
PYTHONPATH=src "$PY" -m edgar_retail_etl.cli ingest --config "$CONFIG"

echo "==> Silver"
PYTHONPATH=src "$PY" -m edgar_retail_etl.cli silver --config "$CONFIG"

echo "==> Gold"
PYTHONPATH=src "$PY" -m edgar_retail_etl.cli gold --config "$CONFIG"

echo "==> Done. Launch dashboard with:"
echo "  $PY -m streamlit run dash/app.py"
