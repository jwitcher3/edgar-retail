VENV=.venv
PY=$(VENV)/bin/python
CONFIG=config.yaml

.PHONY: venv install ingest silver gold run check

venv:
	python -m venv $(VENV)

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

ingest:
	PYTHONPATH=src $(PY) -m edgar_retail_etl.cli ingest --config $(CONFIG)

silver:
	PYTHONPATH=src $(PY) -m edgar_retail_etl.cli silver --config $(CONFIG)

gold:
	PYTHONPATH=src $(PY) -m edgar_retail_etl.cli gold --config $(CONFIG)

run:
	$(PY) -m streamlit run dash/app.py

check:
	$(PY) -c "import typer, duckdb, pandas; print('deps ok')"
