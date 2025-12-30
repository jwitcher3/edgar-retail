from __future__ import annotations

from pathlib import Path
import duckdb
import streamlit as st

st.set_page_config(page_title="EDGAR Retail ETL", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse.duckdb"

st.title("EDGAR Retail: Pressure Signals")

if not DB.exists():
    st.error("DuckDB not found. Run: python -m src.edgar_retail_etl.cli gold (see README)")
    st.stop()

con = duckdb.connect(str(DB), read_only=True)
df = con.execute("SELECT * FROM gold.company_quarter_metrics ORDER BY year DESC, quarter DESC, ticker").df()
con.close()

tickers = sorted(df["ticker"].unique().tolist())
selected = st.multiselect("Tickers", tickers, default=tickers[:5])
dff = df[df["ticker"].isin(selected)].copy()

st.dataframe(dff, width='stretch')
st.bar_chart(dff.set_index("ticker")[["pressure_language_score"]])
