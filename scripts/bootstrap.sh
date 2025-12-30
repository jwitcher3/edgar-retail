#!/usr/bin/env bash
set -euo pipefail

echo "==> Creating repo skeleton (if missing)"
mkdir -p src/edgar_retail_etl dash sql data/{bronze,silver,gold}

touch pyproject.toml requirements.txt .gitignore config.example.yaml README.md \
      src/edgar_retail_etl/__init__.py \
      src/edgar_retail_etl/cli.py \
      src/edgar_retail_etl/settings.py \
      src/edgar_retail_etl/http_client.py \
      src/edgar_retail_etl/edgar_endpoints.py \
      src/edgar_retail_etl/ingest.py \
      src/edgar_retail_etl/parse.py \
      src/edgar_retail_etl/build.py \
      dash/app.py \
      sql/notes.sql

echo "==> Done."
echo "Next: create venv + install deps + copy config:"
echo "  python -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  cp config.example.yaml config.yaml"
