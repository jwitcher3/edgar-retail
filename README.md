# EDGAR Retail ETL (Local-first, VS Code)

## Quickstart

```bash
make venv
make install
make ingest
make pipeline
make app
```


## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.example.yaml config.yaml
# edit config.yaml: sec.user_agent must include contact info
```


## Screenshoots

## Screenshots

### Home (Watchlist)
![Home](docs/images/home.png)

### Company Deep Dive
![Deep Dive](docs/images/deep_dive_1.png)

![Deep Dive](docs/images/deep_dive_2.png)

### Data Quality Checks
![Excerpts](docs/images/data_quality.png)
