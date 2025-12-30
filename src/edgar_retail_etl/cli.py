from __future__ import annotations

from pathlib import Path
import typer

from .settings import load_settings
from .http_client import SecClient
from .ingest import ingest_company_tickers, ticker_to_cik_map, ingest_submissions, ingest_company_facts
from .build import build_silver, build_gold

app = typer.Typer(add_completion=False)

def paths(root: Path):
    data = root / "data"
    return {
        "bronze": data / "bronze",
        "silver": data / "silver",
        "gold": data / "gold",
        "duckdb": data / "warehouse.duckdb",
    }

@app.command("ingest")
def ingest(config: str = typer.Option("config.yaml", "--config")):
    s = load_settings(config)
    p = paths(s.root_dir)

    client = SecClient(
        user_agent=s.sec.user_agent,
        max_rps=s.sec.max_requests_per_second,
        timeout=s.sec.timeout_seconds,
    )

    tickers_pq = ingest_company_tickers(client, p["bronze"])
    cik_map = ticker_to_cik_map(tickers_pq)

    for t in s.project.tickers:
        cik = cik_map.get(t.upper())
        if not cik:
            typer.echo(f"[skip] No CIK for {t}")
            continue
        ingest_submissions(client, cik, p["bronze"])
        ingest_company_facts(client, cik, p["bronze"])

    typer.echo("Done: ingest")

@app.command("silver")
def silver(config: str = typer.Option("config.yaml", "--config")):
    s = load_settings(config)
    if not s.project.tickers:
        raise typer.BadParameter("config.yaml project.tickers is empty.")
    if not s.signals.keywords:
        raise typer.BadParameter("config.yaml signals.keywords is empty.")
    if not s.xbrl.tags:
        raise typer.BadParameter("config.yaml xbrl.tags is empty.")


    p = paths(s.root_dir)

    client = SecClient(
        user_agent=s.sec.user_agent,
        max_rps=s.sec.max_requests_per_second,
        timeout=s.sec.timeout_seconds,
    )

    tickers_pq = p["bronze"] / "company_tickers.parquet"
    if not tickers_pq.exists():
        raise typer.BadParameter("Run ingest first.")

    build_silver(
        client=client,
        bronze_dir=p["bronze"],
        silver_dir=p["silver"],
        tickers_parquet=tickers_pq,
        tickers=s.project.tickers,
        forms=s.project.forms,
        filings_per_company=s.project.filings_per_company,
        keywords=s.signals.keywords,
        tags=s.xbrl.tags,
    )

    typer.echo("Done: silver")

@app.command("gold")
def gold(config: str = typer.Option("config.yaml", "--config")):
    s = load_settings(config)
    p = paths(s.root_dir)

    build_gold(
        silver_dir=p["silver"],
        duckdb_path=p["duckdb"],
        gold_dir=p["gold"],
    )
    typer.echo("Done: gold")

if __name__ == "__main__":
    app()
