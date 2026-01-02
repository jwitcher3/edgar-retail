from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EDGAR Retail — Data Quality", layout="wide")

DB = Path("data/warehouse.duckdb")

st.title("Data Quality")
st.caption("Health checks for the latest pipeline run (run_log + run_warnings).")

if not DB.exists():
    st.error(f"DuckDB not found at: {DB.resolve()}")
    st.info("Run: `make pipeline` (or at least `make gold`) to generate the warehouse.")
    st.stop()

con = duckdb.connect(str(DB), read_only=True)

def table_exists(schema: str, name: str) -> bool:
    q = """
    SELECT COUNT(*) > 0
    FROM information_schema.tables
    WHERE table_schema = ? AND table_name = ?
    """
    return bool(con.execute(q, [schema, name]).fetchone()[0])

if not table_exists("gold", "run_log"):
    st.error("Missing table: gold.run_log")
    st.info("Run: `make gold` again after confirming build.py writes run_log/run_warnings.")
    con.close()
    st.stop()

# Pull run history (latest first)
runs = con.execute("SELECT * FROM gold.run_log ORDER BY run_ts DESC").df()
latest = runs.iloc[0].to_dict()

# Pull warnings for latest run (if table exists)
warnings_df = pd.DataFrame()
if table_exists("gold", "run_warnings"):
    warnings_df = con.execute(
        """
        SELECT ticker, warning_type, detail
        FROM gold.run_warnings
        WHERE run_id = ?
        ORDER BY ticker, warning_type, detail
        """,
        [latest["run_id"]],
    ).df()

con.close()

# --- Top KPI strip ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Run timestamp (UTC)", str(latest.get("run_ts")))
c2.metric("filing_signals rows", int(latest.get("filing_signals_rows", 0)))
c3.metric("xbrl_facts rows", int(latest.get("xbrl_facts_rows", 0)))
c4.metric("pressure_index rows", int(latest.get("pressure_index_rows", 0)))
c5.metric("warnings", int(latest.get("warnings_count", 0)))

st.divider()

# --- Latest run config context (helpful for debugging) ---
with st.expander("Run context (what config was used?)", expanded=False):
    show_cols = [
        "tickers_json",
        "forms_json",
        "filings_per_company",
        "keywords_count",
        "tags_count",
    ]
    ctx = {k: latest.get(k) for k in show_cols}
    st.json(ctx)

# --- Warnings ---
st.subheader("Warnings")

if warnings_df.empty:
    st.success("No warnings for the latest run ✅")
else:
    # Summary by type + ticker
    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Counts by warning type**")
        by_type = warnings_df.groupby("warning_type").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(by_type, width="stretch", hide_index=True)

        st.markdown("**Counts by ticker**")
        by_ticker = warnings_df.groupby("ticker").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(by_ticker, width="stretch", hide_index=True)

    with right:
        st.markdown("**Warning details**")
        st.dataframe(warnings_df, width="stretch", hide_index=True)

        csv = warnings_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download warnings (CSV)",
            data=csv,
            file_name="run_warnings_latest.csv",
            mime="text/csv",
        )

st.divider()

# --- Run history trends ---
st.subheader("Run history")

# Make run_ts friendly
runs2 = runs.copy()
runs2["run_ts"] = pd.to_datetime(runs2["run_ts"], utc=True, errors="coerce")

hist_cols = [
    "run_ts",
    "filing_signals_rows",
    "xbrl_facts_rows",
    "pressure_index_rows",
    "warnings_count",
]
for col in hist_cols[1:]:
    runs2[col] = pd.to_numeric(runs2[col], errors="coerce")

st.dataframe(runs2[hist_cols].head(25), width="stretch", hide_index=True)

st.markdown("**Warnings over time**")
st.line_chart(runs2.set_index("run_ts")["warnings_count"].sort_index())

st.markdown("**Rows over time**")
st.line_chart(
    runs2.set_index("run_ts")[["filing_signals_rows", "xbrl_facts_rows", "pressure_index_rows"]].sort_index()
)

st.divider()

# --- Plain-English guide for juniors ---
st.subheader("What these warnings usually mean (plain English)")

st.markdown(
    """
- **NO_FILINGS_SIGNALS**: We couldn't extract keyword counts from filings for that ticker (often means no filings were found, or HTML download/parse failed).
- **MISSING_XBRL_METRIC**: We couldn't find any of the XBRL tags we expected for that metric (often means the company reports it under a different tag name, or the taxonomy isn't `us-gaap` for that item).
"""
)

st.markdown("**Typical fixes**")
st.markdown(
    """
1) Add more candidate tags for the metric in `config.yaml` (example: add alternative revenue tags).
2) Re-run: `make silver` then `make gold` (or just `make pipeline`).
3) If a company truly doesn’t report the metric in companyfacts, treat it as missing and don’t block the pipeline.
"""
)
