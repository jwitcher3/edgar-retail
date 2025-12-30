from __future__ import annotations

from pathlib import Path
import typer

from .settings import load_settings
from .http_client import SecClient
from .ingest import ingest_company_tickers, ticker_to_cik_map, ingest_submissions, ingest_company_facts, load_ticker_overrides, resolve_cik_for_ticker
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
@app.command("ingest")
def ingest(config: str = typer.Option("config.yaml", "--config")):
    s = load_settings(config)
    p = paths(s.root_dir)

    client = SecClient(
        user_agent=s.sec.user_agent,
        max_rps=s.sec.max_requests_per_second,
        timeout=s.sec.timeout_seconds,
    )

    # Download/refresh mapping parquet (now should include exchange json too)
    tickers_pq = ingest_company_tickers(client, p["bronze"])
    ticker_map = ticker_to_cik_map(tickers_pq)

    # Local override map (data/bronze/ticker_overrides.json)
    overrides = load_ticker_overrides(p["bronze"])

    for t in s.project.tickers:
        cik = resolve_cik_for_ticker(t, ticker_map, overrides=overrides)
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
    if not s.xbrl.metrics:
        raise typer.BadParameter("config.yaml xbrl.metrics is empty.")



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
        tags=s.xbrl.all_tags,
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
        run_context={
            "tickers": s.project.tickers,
            "forms": s.project.forms,
            "filings_per_company": s.project.filings_per_company,
            "keywords": s.signals.keywords,
            "xbrl_metrics": s.xbrl.metrics,
            "xbrl_tags": s.xbrl.all_tags,
        },
    )
    typer.echo("Done: gold")

@app.command("validate")
def validate(config: str = typer.Option("config.yaml", "--config")):
    import duckdb

    s = load_settings(config)
    p = paths(s.root_dir)

    if not p["duckdb"].exists():
        raise typer.BadParameter("DuckDB not found. Run gold first.")

    con = duckdb.connect(str(p["duckdb"]), read_only=True)

    # latest run
    try:
        run = con.execute(
            "SELECT * FROM gold.run_log ORDER BY run_ts DESC LIMIT 1"
        ).fetchdf()
    except Exception:
        con.close()
        typer.echo("No gold.run_log found. Re-run gold after adding run log.")
        raise typer.Exit(code=1)

    r = run.iloc[0].to_dict()
    typer.echo("\n=== Latest run ===")
    typer.echo(f"run_ts (UTC): {r.get('run_ts')}")
    typer.echo(f"filing_signals_rows: {r.get('filing_signals_rows')}")
    typer.echo(f"xbrl_facts_rows: {r.get('xbrl_facts_rows')}")
    typer.echo(f"pressure_index_rows: {r.get('pressure_index_rows')}")
    typer.echo(f"warnings_count: {r.get('warnings_count')}\n")

    # warnings preview
    w = con.execute(
        """
        SELECT ticker, warning_type, detail
        FROM gold.run_warnings
        WHERE run_id = ?
        ORDER BY ticker, warning_type
        LIMIT 50
        """,
        [r["run_id"]],
    ).fetchall()

    if w:
        typer.echo("=== Warnings (up to 50) ===")
        for t, wt, d in w:
            typer.echo(f"- {t}: {wt} ({d})")
    else:
        typer.echo("No warnings ✅")

    con.close()

if __name__ == "__main__":
    app()
