from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .http_client import SecClient
from .ingest import ticker_to_cik_map
from .parse import (
    extract_recent_filings,
    filings_to_df,
    html_to_text,
    compute_keyword_counts,
    primary_doc_url,
)


def build_silver(
    client: SecClient,
    bronze_dir: Path,
    silver_dir: Path,
    tickers_parquet: Path,
    tickers: list[str],
    forms: list[str],
    filings_per_company: int,
    keywords: list[str],
    tags: list[str],
) -> tuple[Path, Path, Path]:
    cik_map = ticker_to_cik_map(tickers_parquet)
    forms_set = set(forms)

    all_filings = []
    all_signals = []
    fact_rows = []

    for t in [x.upper() for x in tickers]:
        cik = cik_map.get(t)
        if not cik:
            continue

        sub_path = bronze_dir / "submissions" / f"CIK{cik:010d}.json"
        facts_path = bronze_dir / "companyfacts" / f"CIK{cik:010d}.json"
        if not sub_path.exists() or not facts_path.exists():
            continue

        submissions = json.loads(sub_path.read_text(encoding="utf-8"))
        filings = extract_recent_filings(submissions, forms_set, filings_per_company)
        df_f = filings_to_df(filings)
        if not df_f.empty:
            df_f.insert(0, "ticker", t)
            all_filings.append(df_f)

        # keyword signals from primary doc html
        for f in filings:
            url = primary_doc_url(f)
            html_cache = bronze_dir / "filings" / t / f"{f.accession}.html"
            html = client.cached_get(url, html_cache, expect="text")
            counts = compute_keyword_counts(html_to_text(html), keywords)
            all_signals.append(
                {
                    "ticker": t,
                    "cik": f.cik,
                    "accession": f.accession,
                    "form": f.form,
                    "filing_date": f.filing_date,
                    "report_date": f.report_date,
                    **counts,
                }
            )

        # XBRL facts (companyfacts)
        doc = json.loads(facts_path.read_text(encoding="utf-8"))
        us_gaap = ((doc.get("facts") or {}).get("us-gaap") or {})
        for tag in tags:
            node = us_gaap.get(tag)
            if not node:
                continue
            for unit, arr in (node.get("units") or {}).items():
                for rec in arr:
                    fact_rows.append(
                        {
                            "ticker": t,
                            "cik": cik,
                            "tag": tag,
                            "unit": unit,
                            "val": rec.get("val"),
                            "end": rec.get("end"),
                            "filed": rec.get("filed"),
                            "form": rec.get("form"),
                            "accn": rec.get("accn"),
                            "fp": rec.get("fp"),
                            "fy": rec.get("fy"),
                        }
                    )

    silver_dir.mkdir(parents=True, exist_ok=True)

    filings_out = silver_dir / "filings.parquet"
    signals_out = silver_dir / "filing_signals.parquet"
    facts_out = silver_dir / "xbrl_facts_long.parquet"

    (pd.concat(all_filings, ignore_index=True) if all_filings else pd.DataFrame()).to_parquet(
        filings_out, index=False
    )
    pd.DataFrame(all_signals).to_parquet(signals_out, index=False)
    pd.DataFrame(fact_rows).to_parquet(facts_out, index=False)

    return filings_out, signals_out, facts_out


def build_gold(silver_dir: Path, duckdb_path: Path, gold_dir: Path) -> tuple[Path, Path]:
    import duckdb

    con = duckdb.connect(str(duckdb_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    con.execute("DROP TABLE IF EXISTS silver.filing_signals;")
    con.execute("DROP TABLE IF EXISTS silver.xbrl_facts_long;")

    con.execute(
        "CREATE TABLE silver.filing_signals AS SELECT * FROM read_parquet(?);",
        [str(silver_dir / "filing_signals.parquet")],
    )
    con.execute(
        "CREATE TABLE silver.xbrl_facts_long AS SELECT * FROM read_parquet(?);",
        [str(silver_dir / "xbrl_facts_long.parquet")],
    )

    con.execute("DROP TABLE IF EXISTS gold.quarter_facts;")
    con.execute(
        """
        CREATE TABLE gold.quarter_facts AS
        WITH q AS (
            SELECT
                ticker,
                tag,
                val::DOUBLE AS val,
                CAST("end" AS DATE) AS end_date
            FROM silver.xbrl_facts_long
            WHERE "end" IS NOT NULL
        ),
        q2 AS (
            SELECT
                ticker,
                tag,
                val,
                EXTRACT(year FROM end_date) AS year,
                EXTRACT(quarter FROM end_date) AS quarter
            FROM q
        )
        SELECT
            ticker, year, quarter,
            MAX(CASE WHEN tag='InventoryNet' THEN val END) AS inventory_net,
            MAX(CASE WHEN tag='NetSales' THEN val END) AS net_sales,
            MAX(CASE WHEN tag='GrossProfit' THEN val END) AS gross_profit,
            MAX(CASE WHEN tag='OperatingIncomeLoss' THEN val END) AS op_income_loss
        FROM q2
        GROUP BY 1,2,3;
        """
    )

    con.execute("DROP TABLE IF EXISTS gold.company_quarter_metrics;")
    con.execute(
        """
        CREATE TABLE gold.company_quarter_metrics AS
        WITH s AS (
          SELECT
            ticker,
            COALESCE(report_date, filing_date) AS dt,
            kw_inventory, kw_promotion, kw_promotional, kw_markdown,
            kw_demand, kw_traffic, kw_pricing, kw_guidance
          FROM silver.filing_signals
        ),
        sq AS (
          SELECT
            ticker,
            EXTRACT(year FROM CAST(dt AS DATE)) AS year,
            EXTRACT(quarter FROM CAST(dt AS DATE)) AS quarter,
            MAX(kw_inventory) AS kw_inventory,
            MAX(kw_promotion) AS kw_promotion,
            MAX(kw_promotional) AS kw_promotional,
            MAX(kw_markdown) AS kw_markdown,
            MAX(kw_demand) AS kw_demand,
            MAX(kw_traffic) AS kw_traffic,
            MAX(kw_pricing) AS kw_pricing,
            MAX(kw_guidance) AS kw_guidance
          FROM s
          GROUP BY 1,2,3
        )
        SELECT
          sq.*,
          qf.inventory_net, qf.net_sales, qf.gross_profit, qf.op_income_loss,
          COALESCE(sq.kw_inventory,0) + COALESCE(sq.kw_promotion,0) + COALESCE(sq.kw_markdown,0)
            AS pressure_language_score
        FROM sq
        LEFT JOIN gold.quarter_facts qf
          ON sq.ticker=qf.ticker AND sq.year=qf.year AND sq.quarter=qf.quarter;
        """
    )

    gold_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = gold_dir / "company_quarter_metrics.parquet"
    out_alerts = gold_dir / "company_quarter_alerts.parquet"

    con.execute("COPY gold.company_quarter_metrics TO ? (FORMAT PARQUET);", [str(out_metrics)])
    con.execute("COPY gold.company_quarter_metrics TO ? (FORMAT PARQUET);", [str(out_alerts)])

    con.close()
    return out_metrics, out_alerts
