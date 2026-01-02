from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .http_client import SecClient
from .ingest import ticker_to_cik_map
from .parse import extract_recent_filings, filings_to_df, html_to_text, compute_keyword_counts, primary_doc_url



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

    all_filings: list[pd.DataFrame] = []
    all_signals: list[dict] = []
    fact_rows: list[dict] = []
    filing_text_rows: list[dict] = []

    # stable schemas (prevents "no columns" parquet issues)
    filings_cols = ["ticker", "cik", "accession", "form", "filing_date", "report_date", "primary_doc"]
    signals_cols = (
        ["ticker", "cik", "accession", "form", "filing_date", "report_date"]
        + [f"kw_{k.lower()}" for k in keywords]
    )
    facts_cols = ["ticker", "cik", "tag", "unit", "val", "end", "filed", "form", "accn", "fp", "fy"]
    filing_text_cols = [
        "ticker", "cik", "accession", "form",
        "filing_date", "report_date", "dt",
        "url", "text", "text_len"
    ]

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

        # keyword signals + filing text (primary doc)
        for f in filings:
            url = primary_doc_url(f)
            html_cache = bronze_dir / "filings" / t / f"{f.accession}.html"
            html = client.cached_get(url, html_cache, expect="text")

            text = html_to_text(html)
            counts = compute_keyword_counts(text, keywords)

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

            dt = f.report_date or f.filing_date
            filing_text_rows.append(
                {
                    "ticker": t,
                    "cik": f.cik,
                    "accession": f.accession,
                    "form": f.form,
                    "filing_date": f.filing_date,
                    "report_date": f.report_date,
                    "dt": dt,
                    "url": url,
                    "text": text,
                    "text_len": len(text) if text is not None else 0,
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
    filing_text_out = silver_dir / "filing_text.parquet"

    # Write schema-safe outputs
    df_filings = pd.concat(all_filings, ignore_index=True) if all_filings else pd.DataFrame(columns=filings_cols)
    df_filings = df_filings.reindex(columns=filings_cols)
    df_filings.to_parquet(filings_out, index=False)

    df_signals = pd.DataFrame(all_signals) if all_signals else pd.DataFrame(columns=signals_cols)
    df_signals = df_signals.reindex(columns=signals_cols)
    df_signals.to_parquet(signals_out, index=False)

    df_facts = pd.DataFrame(fact_rows) if fact_rows else pd.DataFrame(columns=facts_cols)
    df_facts = df_facts.reindex(columns=facts_cols)
    df_facts.to_parquet(facts_out, index=False)

    df_text = pd.DataFrame(filing_text_rows) if filing_text_rows else pd.DataFrame(columns=filing_text_cols)
    df_text = df_text.reindex(columns=filing_text_cols)
    df_text.to_parquet(filing_text_out, index=False)

    return filings_out, signals_out, facts_out



def build_gold(
    silver_dir: Path,
    duckdb_path: Path,
    gold_dir: Path,
    run_context: dict | None = None,
) -> tuple[Path, Path]:



    import duckdb

    con = duckdb.connect(str(duckdb_path))

    def _coalesce_max(tags: list[str], alias: str) -> str:
        if not tags:
            return f"NULL::DOUBLE AS {alias}"
        parts = [f"MAX(CASE WHEN tag='{t}' THEN val END)" for t in tags]
        return "COALESCE(" + ", ".join(parts) + f") AS {alias}"         

    con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")

    con.execute("DROP TABLE IF EXISTS silver.filing_signals;")
    con.execute("DROP TABLE IF EXISTS silver.xbrl_facts_long;")
    con.execute("DROP TABLE IF EXISTS silver.filing_text;")  # NEW

    con.execute(
        "CREATE TABLE silver.filing_signals AS SELECT * FROM read_parquet(?);",
        [str(silver_dir / "filing_signals.parquet")],
    )
    con.execute(
        "CREATE TABLE silver.xbrl_facts_long AS SELECT * FROM read_parquet(?);",
        [str(silver_dir / "xbrl_facts_long.parquet")],
    )
    con.execute(  # NEW
        "CREATE TABLE silver.filing_text AS SELECT * FROM read_parquet(?);",
        [str(silver_dir / "filing_text.parquet")],
    )


    con.execute("DROP TABLE IF EXISTS gold.quarter_facts;")

    ctx = run_context or {}
    xbrl_metrics = ctx.get("xbrl_metrics") or {}

    inv_tags = list(xbrl_metrics.get("inventory", []))
    rev_tags = list(xbrl_metrics.get("revenue", []))
    cogs_tags = list(xbrl_metrics.get("cogs", []))    
    gp_tags = list(xbrl_metrics.get("gross_profit", []))
    op_tags = list(xbrl_metrics.get("op_income", []))
    
    quarter_facts_sql = f"""
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
    ),
    agg AS (
    SELECT
        ticker,
        year,
        quarter,
        {_coalesce_max(inv_tags, "inventory")},
        {_coalesce_max(rev_tags, "revenue")},
        {_coalesce_max(cogs_tags, "cogs")},
        {_coalesce_max(gp_tags, "gross_profit_raw")},
        {_coalesce_max(op_tags, "op_income")}
    FROM q2
    GROUP BY 1,2,3
    )
    SELECT
    ticker,
    year,
    quarter,
    inventory,
    revenue,
    cogs,
    COALESCE(
        gross_profit_raw,
        CASE
        WHEN revenue IS NOT NULL AND cogs IS NOT NULL THEN revenue - cogs
        ELSE NULL
        END
    ) AS gross_profit,
    op_income
    FROM agg;
    """

    con.execute(quarter_facts_sql)

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
          qf.inventory, qf.revenue, qf.cogs, qf.gross_profit, qf.op_income,
          COALESCE(sq.kw_inventory,0) + COALESCE(sq.kw_promotion,0) + COALESCE(sq.kw_markdown,0)
            AS pressure_language_score
        FROM sq
        LEFT JOIN gold.quarter_facts qf
          ON sq.ticker=qf.ticker AND sq.year=qf.year AND sq.quarter=qf.quarter;
        """
    )

    con.execute("DROP TABLE IF EXISTS gold.company_quarter_features;")
    con.execute(
        """
        CREATE TABLE gold.company_quarter_features AS
        WITH base AS (
          SELECT
            *,
            CASE
              WHEN revenue IS NOT NULL AND revenue != 0 AND inventory IS NOT NULL
              THEN inventory / revenue
              ELSE NULL
            END AS inventory_to_sales
          FROM gold.company_quarter_metrics
        ),
        lagged AS (
          SELECT
            *,
            LAG(inventory) OVER (PARTITION BY ticker ORDER BY year, quarter) AS inventory_net_prev_q,
            LAG(revenue) OVER (PARTITION BY ticker ORDER BY year, quarter) AS net_sales_prev_q,
            LAG(pressure_language_score) OVER (PARTITION BY ticker ORDER BY year, quarter) AS pressure_prev_q,
            LAG(inventory_to_sales) OVER (PARTITION BY ticker ORDER BY year, quarter) AS inv_to_sales_prev_q
          FROM base
        )
        SELECT
          *,
          CASE
            WHEN inventory_net_prev_q IS NOT NULL AND inventory_net_prev_q != 0 AND inventory IS NOT NULL
            THEN (inventory - inventory_net_prev_q) / inventory_net_prev_q
            ELSE NULL
          END AS inventory_net_qoq_pct,
          CASE
            WHEN net_sales_prev_q IS NOT NULL AND net_sales_prev_q != 0 AND revenue IS NOT NULL
            THEN (revenue - net_sales_prev_q) / net_sales_prev_q
            ELSE NULL
          END AS net_sales_qoq_pct,
          CASE
            WHEN pressure_prev_q IS NOT NULL AND pressure_prev_q != 0 AND pressure_language_score IS NOT NULL
            THEN (pressure_language_score - pressure_prev_q) * 1.0 / pressure_prev_q
            ELSE NULL
          END AS pressure_qoq_pct,
          CASE
            WHEN inv_to_sales_prev_q IS NOT NULL AND inv_to_sales_prev_q != 0 AND inventory_to_sales IS NOT NULL
            THEN (inventory_to_sales - inv_to_sales_prev_q) / inv_to_sales_prev_q
            ELSE NULL
          END AS inventory_to_sales_qoq_pct
        FROM lagged;
        """
    )

    con.execute("DROP TABLE IF EXISTS gold.pressure_index;")
    con.execute(
        """
        CREATE TABLE gold.pressure_index AS
        WITH f AS (
          SELECT
            ticker, year, quarter,
            pressure_language_score,
            inventory_to_sales
          FROM gold.company_quarter_features
        ),
        stats AS (
          SELECT
            year, quarter,
            AVG(pressure_language_score) AS m_pressure,
            STDDEV_SAMP(pressure_language_score) AS s_pressure,
            AVG(inventory_to_sales) AS m_inv2sales,
            STDDEV_SAMP(inventory_to_sales) AS s_inv2sales
          FROM f
          GROUP BY 1,2
        )
        SELECT
          f.ticker,
          f.year,
          f.quarter,
          f.pressure_language_score,
          f.inventory_to_sales,
          CASE
            WHEN stats.s_pressure IS NOT NULL AND stats.s_pressure != 0 AND f.pressure_language_score IS NOT NULL
            THEN (f.pressure_language_score - stats.m_pressure) / stats.s_pressure
            ELSE NULL
          END AS z_pressure_language,
          CASE
            WHEN stats.s_inv2sales IS NOT NULL AND stats.s_inv2sales != 0 AND f.inventory_to_sales IS NOT NULL
            THEN (f.inventory_to_sales - stats.m_inv2sales) / stats.s_inv2sales
            ELSE NULL
          END AS z_inventory_to_sales,
          -- Simple composite index (equal weight)
          CASE
            WHEN
              (stats.s_pressure IS NOT NULL AND stats.s_pressure != 0 AND f.pressure_language_score IS NOT NULL)
              OR
              (stats.s_inv2sales IS NOT NULL AND stats.s_inv2sales != 0 AND f.inventory_to_sales IS NOT NULL)
            THEN
              COALESCE((f.pressure_language_score - stats.m_pressure) / NULLIF(stats.s_pressure,0), 0)
              +
              COALESCE((f.inventory_to_sales - stats.m_inv2sales) / NULLIF(stats.s_inv2sales,0), 0)
            ELSE NULL
          END AS pressure_index
        FROM f
        JOIN stats USING (year, quarter);
        """
    )
    # ---------------------------
    # Run log + data quality checks
    # ---------------------------
    run_id = str(uuid.uuid4())
    run_ts = datetime.now(timezone.utc)

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold.run_log (
          run_id VARCHAR,
          run_ts TIMESTAMPTZ,
          tickers_json VARCHAR,
          forms_json VARCHAR,
          filings_per_company INTEGER,
          keywords_count INTEGER,
          tags_count INTEGER,
          filing_signals_rows BIGINT,
          xbrl_facts_rows BIGINT,
          pressure_index_rows BIGINT,
          warnings_count BIGINT
        );
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS gold.run_warnings (
          run_id VARCHAR,
          ticker VARCHAR,
          warning_type VARCHAR,
          detail VARCHAR
        );
        """
    )

    # Counts from silver/gold
    filing_signals_rows = con.execute("SELECT COUNT(*) FROM silver.filing_signals").fetchone()[0]
    xbrl_facts_rows = con.execute("SELECT COUNT(*) FROM silver.xbrl_facts_long").fetchone()[0]

    # pressure_index may not exist if you haven't created it yet in your build
    pressure_index_rows = 0
    try:
        pressure_index_rows = con.execute("SELECT COUNT(*) FROM gold.pressure_index").fetchone()[0]
    except Exception:
        pressure_index_rows = 0

    ctx = run_context or {}
    tickers = [str(t).upper() for t in ctx.get("tickers", [])]
    xbrl_metrics = ctx.get("xbrl_metrics") or {}
    tags = [str(x) for x in ctx.get("tags", [])]
    forms = [str(x) for x in ctx.get("forms", [])]
    filings_per_company = int(ctx.get("filings_per_company", 0) or 0)
    keywords_count = int(len(ctx.get("keywords", []) or []))
    tags_count = int(len(tags))

    # Which tickers had any filing signals?
    tickers_with_signals = set(
        r[0]
        for r in con.execute("SELECT DISTINCT ticker FROM silver.filing_signals").fetchall()
    )

    # Which (ticker, tag) pairs exist?
    facts_pairs = set(
        (r[0], r[1])
        for r in con.execute("SELECT DISTINCT ticker, tag FROM silver.xbrl_facts_long").fetchall()
    )

    warnings = []

    # Missing signals
    for t in tickers:
        if t not in tickers_with_signals:
            warnings.append((run_id, t, "NO_FILINGS_SIGNALS", "No filing_signals rows for ticker"))

    for t in tickers:
        if t not in tickers_with_signals:
            continue

        for metric_name, candidate_tags in xbrl_metrics.items():
            # derived gross profit: allow (revenue + cogs) to satisfy even if GrossProfit absent
            if metric_name == "gross_profit":
                rev_tags = xbrl_metrics.get("revenue", [])
                cogs_tags = xbrl_metrics.get("cogs", [])
                has_rev = any((t, tag) in facts_pairs for tag in rev_tags)
                has_cogs = any((t, tag) in facts_pairs for tag in cogs_tags)
                has_gp = any((t, tag) in facts_pairs for tag in candidate_tags)
                has_any = has_gp or (has_rev and has_cogs)
            else:
                has_any = any((t, tag) in facts_pairs for tag in candidate_tags)

            if not has_any:
                warnings.append(
                    (run_id, t, "MISSING_XBRL_METRIC", f"{metric_name}: {candidate_tags}")
                )

    # Deduplicate warnings (stable)
    warnings = list(dict.fromkeys(warnings))

    if warnings:
        con.executemany(
            "INSERT INTO gold.run_warnings (run_id, ticker, warning_type, detail) VALUES (?, ?, ?, ?)",
            warnings,
        )

    warnings_count = len(warnings)

    con.execute(
        """
        INSERT INTO gold.run_log (
          run_id, run_ts, tickers_json, forms_json, filings_per_company,
          keywords_count, tags_count,
          filing_signals_rows, xbrl_facts_rows, pressure_index_rows, warnings_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            run_ts,
            json.dumps(tickers),
            json.dumps(forms),
            filings_per_company,
            keywords_count,
            tags_count,
            filing_signals_rows,
            xbrl_facts_rows,
            pressure_index_rows,
            warnings_count,
        ],
    )

    # Export run log snapshots to gold/ as Parquet for quick inspection
    gold_dir.mkdir(parents=True, exist_ok=True)
    con.execute(
        "COPY (SELECT * FROM gold.run_log ORDER BY run_ts DESC) TO ? (FORMAT PARQUET);",
        [str(gold_dir / "run_log.parquet")],
    )
    con.execute(
        "COPY (SELECT * FROM gold.run_warnings) TO ? (FORMAT PARQUET);",
        [str(gold_dir / "run_warnings.parquet")],
    )




    gold_dir.mkdir(parents=True, exist_ok=True)
    out_metrics = gold_dir / "company_quarter_metrics.parquet"
    out_alerts = gold_dir / "company_quarter_alerts.parquet"

    con.execute("COPY gold.company_quarter_metrics TO ? (FORMAT PARQUET);", [str(out_metrics)])
    con.execute("COPY gold.company_quarter_metrics TO ? (FORMAT PARQUET);", [str(out_alerts)])

    con.close()
    return out_metrics, out_alerts
