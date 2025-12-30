from __future__ import annotations

from pathlib import Path
import pandas as pd

from .http_client import SecClient
from .edgar_endpoints import COMPANY_TICKERS_JSON, SUBMISSIONS_JSON, COMPANY_FACTS_JSON


def ingest_company_tickers(client: SecClient, bronze_dir: Path) -> Path:
    out_json = bronze_dir / "company_tickers.json"
    data = client.cached_get(COMPANY_TICKERS_JSON, out_json, expect="json")

    rows = []
    for _, rec in data.items():
        rows.append({"cik": int(rec["cik_str"]), "ticker": rec["ticker"], "title": rec["title"]})

    df = pd.DataFrame(rows)
    out_pq = bronze_dir / "company_tickers.parquet"
    df.to_parquet(out_pq, index=False)
    return out_pq


def ticker_to_cik_map(company_tickers_parquet: Path) -> dict[str, int]:
    df = pd.read_parquet(company_tickers_parquet)
    return {str(t).upper(): int(c) for t, c in zip(df["ticker"], df["cik"])}


def ingest_submissions(client: SecClient, cik: int, bronze_dir: Path) -> Path:
    cik10 = f"{cik:010d}"
    url = SUBMISSIONS_JSON.format(cik10=cik10)
    out = bronze_dir / "submissions" / f"CIK{cik10}.json"
    client.cached_get(url, out, expect="json")
    return out


def ingest_company_facts(client: SecClient, cik: int, bronze_dir: Path) -> Path:
    cik10 = f"{cik:010d}"
    url = COMPANY_FACTS_JSON.format(cik10=cik10)
    out = bronze_dir / "companyfacts" / f"CIK{cik10}.json"
    client.cached_get(url, out, expect="json")
    return out
