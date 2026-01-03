VENV=.venv
PY=$(VENV)/bin/python
CONFIG=config.yaml
ENV=PYTHONPATH=src

.PHONY: venv install ingest silver gold validate pipeline app check help

venv:
	python -m venv $(VENV)

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	$(ENV) $(PY) -m edgar_retail_etl.cli ingest --config $(CONFIG)

silver:
	$(ENV) $(PY) -m edgar_retail_etl.cli silver --config $(CONFIG)

gold:
	$(ENV) $(PY) -m edgar_retail_etl.cli gold --config $(CONFIG)

validate:
	$(ENV) $(PY) -m edgar_retail_etl.cli validate --config $(CONFIG)

# Full refresh after you've ingested at least once
pipeline: silver gold validate

# Run dashboard only
app:
	$(PY) -m streamlit run dash/app.py

check:
	$(PY) -c "import typer, duckdb, pandas; print('deps ok')"

help:
	@echo ""
	@echo "Targets:"
	@echo "  make venv       - create virtual environment"
	@echo "  make install    - install dependencies"
	@echo "  make ingest     - download raw SEC data into data/bronze"
	@echo "  make silver     - build silver parquet outputs"
	@echo "  make gold       - build DuckDB + gold outputs"
	@echo "  make validate   - show latest run + warnings"
	@echo "  make pipeline   - silver + gold + validate"
	@echo "  make app        - launch Streamlit dashboard"
	@echo "  make check      - quick import check"
	@echo ""

smoke:
	$(ENV) $(PY) -m edgar_retail_etl.cli silver --config $(CONFIG)
	$(ENV) $(PY) -m edgar_retail_etl.cli gold --config $(CONFIG)
	$(ENV) $(PY) -m edgar_retail_etl.cli validate --config $(CONFIG)
