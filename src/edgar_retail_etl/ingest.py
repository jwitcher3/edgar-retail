from __future__ import annotations

from pathlib import Path
import pandas as pd
import json

from .http_client import SecClient
from .edgar_endpoints import COMPANY_TICKERS_JSON, SUBMISSIONS_JSON, COMPANY_FACTS_JSON


from .edgar_endpoints import (
    COMPANY_TICKERS_JSON,
    COMPANY_TICKERS_EXCHANGE_JSON,
    SUBMISSIONS_JSON,
    COMPANY_FACTS_JSON,
)

def ingest_company_tickers(client: SecClient, bronze_dir: Path) -> Path:
    bronze_dir.mkdir(parents=True, exist_ok=True)

    # Download both JSON files to bronze/
    out_json_base = bronze_dir / "company_tickers.json"
    out_json_exch = bronze_dir / "company_tickers_exchange.json"

    client.cached_get(COMPANY_TICKERS_JSON, out_json_base, expect="json")
    client.cached_get(COMPANY_TICKERS_EXCHANGE_JSON, out_json_exch, expect="json")

    # Build a merged ticker->cik map (exchange overrides base)
    ticker_map: dict[str, int] = {}
    if out_json_base.exists():
        ticker_map.update(_load_ticker_map(out_json_base))
    if out_json_exch.exists():
        ticker_map.update(_load_ticker_map(out_json_exch))
    # Local overrides win (e.g., FL / SKX)
    overrides = load_ticker_overrides(bronze_dir)
    ticker_map.update(overrides)
    # Write a single parquet for downstream use
    rows = [{"ticker": t, "cik": cik} for t, cik in sorted(ticker_map.items())]
    df = pd.DataFrame(rows)
    out_pq = bronze_dir / "company_tickers.parquet"
    df.to_parquet(out_pq, index=False)
    return out_pq

def load_ticker_overrides(bronze_dir: Path) -> dict[str, int]:
    p = bronze_dir / "ticker_overrides.json"
    if not p.exists():
        return {}
    obj = json.loads(p.read_text())
    return {str(k).upper(): int(v) for k, v in obj.items()}

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

def _load_ticker_map(path: Path) -> dict[str, int]:
    j = json.loads(path.read_text())

    out: dict[str, int] = {}

    # Format A: {"fields":[...], "data":[[...], ...]}
    if isinstance(j, dict) and "fields" in j and "data" in j:
        fields = j["fields"]
        idx = {f: i for i, f in enumerate(fields)}
        for row in j["data"]:
            try:
                t = str(row[idx["ticker"]]).upper()
                cik = int(row[idx["cik"]])
                if t:
                    out[t] = cik
            except Exception:
                continue
        return out

    # Format B: dict or list of dict rows
    vals = j.values() if isinstance(j, dict) else j
    for r in vals:
        if not isinstance(r, dict):
            continue
        t = (r.get("ticker") or "").upper()
        cik = r.get("cik") or r.get("cik_str")
        if t and cik:
            out[t] = int(cik)

    return out


def resolve_cik_for_ticker(
    ticker: str,
    ticker_map: dict[str, int],
    overrides: dict[str, int] | None = None,
) -> int | None:
    t = ticker.upper().strip()

    if overrides and t in overrides:
        return overrides[t]

    # direct
    if t in ticker_map:
        return ticker_map[t]

    # common normalization: remove punctuation and class suffixes
    t_norm = t.replace(".", "").replace("-", "").replace("/", "")
    for k, v in ticker_map.items():
        k_norm = k.replace(".", "").replace("-", "").replace("/", "")
        if k_norm == t_norm:
            return v

    # last resort: startswith match (rare but helps)
    for k, v in ticker_map.items():
        if k.startswith(t):
            return v

    return None

