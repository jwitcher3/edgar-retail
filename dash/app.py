from __future__ import annotations

from pathlib import Path
import duckdb
import streamlit as st

st.set_page_config(page_title="EDGAR Retail ETL", layout="wide")

with st.sidebar:
    st.markdown("### James Witcher")
    st.caption("Marketing / Product Analytics • Data Science")
    st.markdown("✉️ [james.witcher@outlook.com](mailto:james.witcher@outlook.com)")
    st.markdown("🔗 [GitHub](https://github.com/jwitcher3)  •  [LinkedIn](https://www.linkedin.com/in/james-witcher/)")
    st.divider()


st.title("EDGAR Retail: Pressure Signals")
st.caption("Quarterly pressure signals + financial context pulled from SEC EDGAR filings (DuckDB gold tables).")

with st.expander("How to use this app", expanded=True):
    st.markdown(
        """
1) **Start here (Home):** pick a few tickers and scan pressure levels + trends.  
2) **Company Deep Dive:** choose a ticker → choose a quarter → review:
   - pressure_index + components
   - keyword driver mix (inventory / promotion / demand / pricing / guidance)
   - top excerpts across filings for that quarter (with links)
3) **Exports:** download quarter bundles (CSV / MD / PDF / ZIP) for sharing.
        """
    )

col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/2_Company_Deep_Dive.py", label="➡️ Open Company Deep Dive", icon="📊")
with col2:
    st.page_link("pages/02_Data_Quality.py", label="➡️ Open Data Quality", icon="✅")


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse.duckdb"

st.title("EDGAR Retail: Pressure Signals")

if not DB.exists():
    st.error("DuckDB not found. Run: python -m src.edgar_retail_etl.cli gold (see README)")
    st.stop()

con = duckdb.connect(str(DB), read_only=True)

st.subheader("Latest run")

try:
    run = con.execute("SELECT * FROM gold.run_log ORDER BY run_ts DESC LIMIT 1").df()
    st.dataframe(run, width="stretch")
    rid = run.loc[0, "run_id"]
    warns = con.execute(
        "SELECT ticker, warning_type, detail FROM gold.run_warnings WHERE run_id = ? ORDER BY ticker, warning_type",
        [rid],
    ).df()
    if len(warns) > 0:
        st.warning("Warnings found")
        st.dataframe(warns, width="stretch")
    else:
        st.success("No warnings")
except Exception as e:
    st.info("Run log not available yet. Re-run gold to generate gold.run_log.")


df = con.execute(
    "SELECT * FROM gold.pressure_index ORDER BY year DESC, quarter DESC, ticker"
).df()
con.close()

df["period"] = df["year"].astype(str) + " Q" + df["quarter"].astype(str)
df["sort_key"] = df["year"] * 10 + df["quarter"]
df = df.sort_values(["sort_key", "ticker"])

tickers = sorted(df["ticker"].unique().tolist())
selected = st.multiselect("Tickers", tickers, default=tickers[:5])
dff = df[df["ticker"].isin(selected)].copy()

st.dataframe(dff, width='stretch')
st.bar_chart(dff.set_index("ticker")[["pressure_language_score"]])

chart_df = dff.pivot_table(
    index="period",
    columns="ticker",
    values="pressure_index",
    aggfunc="mean"
)
st.line_chart(chart_df)
