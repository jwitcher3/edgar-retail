from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import streamlit as st
import re
import html
import hashlib
from difflib import SequenceMatcher

def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_into_segments(text: str, max_len: int = 650, min_len: int = 80) -> list[str]:
    """
    Split filing text into readable segments.
    1) paragraphs (blank-line separated)
    2) if a paragraph is very long, chunk it by sentence-ish boundaries
    """
    text = _normalize_ws(text)
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    segs: list[str] = []

    sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

    for p in paras:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < min_len:
            continue

        if len(p) <= max_len:
            segs.append(p)
            continue

        # Chunk long paragraph into ~max_len blocks using sentence-ish splits
        sentences = sentence_split.split(p)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_len:
                buf = buf + " " + s
            else:
                if len(buf) >= min_len:
                    segs.append(buf)
                buf = s
        if buf and len(buf) >= min_len:
            segs.append(buf)

    # De-dupe exact duplicates
    seen = set()
    out = []
    for s in segs:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def score_segment(seg: str, terms: list[str]) -> tuple[float, dict[str, int]]:
    """
    Simple, robust scoring:
      - counts occurrences of each term (case-insensitive)
      - bonus for multiple distinct terms in same segment
      - mild length preference (best around ~240 chars)
    """
    lo = seg.lower()
    hits: dict[str, int] = {}
    raw = 0.0

    for t in terms:
        t2 = t.strip()
        if not t2:
            continue
        c = len(re.findall(re.escape(t2.lower()), lo))
        if c:
            hits[t2] = c
            # base + repeats
            raw += 3.0 + 2.0 * (c - 1)

    if not hits:
        return 0.0, {}

    # bonus for multiple distinct terms
    if len(hits) >= 2:
        raw += 2.0 * (len(hits) - 1)

    # length normalization: prefer ~240 chars, but don’t kill longer segments
    target = 240.0
    length = float(len(seg))
    length_factor = 1.0 / (1.0 + abs(length - target) / target)

    return raw * length_factor, hits

def _is_too_similar(a: str, b: str, thresh: float = 0.90) -> bool:
    # Cheap near-duplicate filter
    return SequenceMatcher(None, a, b).ratio() >= thresh

def top_excerpts(text: str, terms: list[str], top_n: int = 10) -> list[dict]:
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        return []

    segs = split_into_segments(text)
    scored = []
    for seg in segs:
        score, hits = score_segment(seg, terms)
        if score > 0:
            scored.append((score, hits, seg))

    scored.sort(key=lambda x: x[0], reverse=True)

    picked: list[dict] = []
    for score, hits, seg in scored:
        # near-duplicate suppression
        if any(_is_too_similar(seg, p["text"]) for p in picked):
            continue
        picked.append({"score": score, "hits": hits, "text": seg})
        if len(picked) >= top_n:
            break
    return picked

def highlight_terms(text: str, terms: list[str]) -> str:
    """
    Safely highlight terms using <mark>, while HTML-escaping everything else.
    """
    terms_sorted = [t for t in sorted({t.strip() for t in terms if t and t.strip()}, key=len, reverse=True)]
    tmp = text

    # Insert marker tokens on original text (preserves original casing),
    # then HTML-escape, then swap tokens for <mark> tags.
    for i, t in enumerate(terms_sorted):
        tmp = re.sub(
            re.escape(t),
            lambda m, i=i: f"[[[MARK{i}]]]{m.group(0)}[[[ENDMARK{i}]]]",
            tmp,
            flags=re.IGNORECASE,
        )

    tmp = html.escape(tmp)

    for i, _ in enumerate(terms_sorted):
        tmp = tmp.replace(f"[[[MARK{i}]]]", "<mark>").replace(f"[[[ENDMARK{i}]]]", "</mark>")

    return tmp.replace("\n", " ")

@st.cache_data(show_spinner=False)
def cached_top_excerpts(text_hash: str, terms_tuple: tuple[str, ...], top_n: int, text: str):
    return top_excerpts(text, list(terms_tuple), top_n)

# --- Integration (put inside your Filing Explorer section, after you have `txt` and `selected_terms`) ----
# Example:
# txt = ...  # full filing text
# selected_terms = ...  # list of user-selected driver terms

# --- End C3 ------------------------------------------------------------------

st.set_page_config(page_title="Company Deep Dive", layout="wide")

ROOT = Path(__file__).resolve().parents[2]  # dash/pages -> repo root
DB = ROOT / "data" / "warehouse.duckdb"

st.title("Company Deep Dive")
st.caption("Quarterly pressure signals + financial context pulled from SEC EDGAR (DuckDB gold tables).")

if not DB.exists():
    st.error("DuckDB not found. Run: `make pipeline` (or `make gold`) first.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_filing_text(db_path: Path, ticker: str, accession: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT ticker, accession, form, filing_date, report_date, dt, url, text, text_len
            FROM silver.filing_text
            WHERE ticker = ? AND accession = ?
            LIMIT 1
            """,
            [ticker, accession],
        ).df()
    finally:
        con.close()

def extract_snippets(text: str, term: str, window: int = 160, max_hits: int = 3) -> list[str]:
    if not text or not term:
        return []
    lower = text.lower()
    term_l = term.lower()
    hits = [m.start() for m in re.finditer(re.escape(term_l), lower)]
    out = []
    for start in hits[:max_hits]:
        end = start + len(term_l)
        left = max(0, start - window)
        right = min(len(text), end + window)
        out.append(text[left:right].replace("\n", " "))
    return out


@st.cache_data(show_spinner=False)
def load_tickers(db_path: Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tickers = con.execute(
            "SELECT DISTINCT ticker FROM gold.pressure_index ORDER BY ticker"
        ).fetchall()
        return [t[0] for t in tickers]
    finally:
        con.close()

@st.cache_data(show_spinner=False)
def load_company_quarters(db_path: Path, ticker: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            WITH base AS (
              SELECT
                pi.ticker, pi.year, pi.quarter,
                pi.pressure_index,
                pi.z_pressure_language,
                pi.z_inventory_to_sales,
                cqm.pressure_language_score,
                cqf.inventory_to_sales,
                cqf.inventory_net_qoq_pct,
                cqf.net_sales_qoq_pct,
                cqf.pressure_qoq_pct,
                cqf.inventory_to_sales_qoq_pct,
                cqm.inventory, cqm.revenue, cqm.cogs, cqm.gross_profit, cqm.op_income,
                cqm.kw_inventory, cqm.kw_promotion, cqm.kw_promotional, cqm.kw_markdown,
                cqm.kw_demand, cqm.kw_traffic, cqm.kw_pricing, cqm.kw_guidance
              FROM gold.pressure_index pi
              LEFT JOIN gold.company_quarter_metrics cqm
                USING (ticker, year, quarter)
              LEFT JOIN gold.company_quarter_features cqf
                USING (ticker, year, quarter)
              WHERE pi.ticker = ?
            )
            SELECT * FROM base
            ORDER BY year, quarter
            """,
            [ticker],
        ).df()

        if df.empty:
            return df

        df["period"] = df["year"].astype(int).astype(str) + " Q" + df["quarter"].astype(int).astype(str)
        df["sort_key"] = df["year"].astype(int) * 10 + df["quarter"].astype(int)
        return df
    finally:
        con.close()

@st.cache_data(show_spinner=False)
def load_recent_filings(db_path: Path, ticker: str) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        # filing-level signals
        df = con.execute(
            """
            SELECT
              ticker,
              accession,
              form,
              filing_date,
              report_date,
              COALESCE(report_date, filing_date) AS dt,
              kw_inventory, kw_promotion, kw_promotional, kw_markdown,
              kw_demand, kw_traffic, kw_pricing, kw_guidance
            FROM silver.filing_signals
            WHERE ticker = ?
            ORDER BY CAST(COALESCE(report_date, filing_date) AS DATE) DESC
            LIMIT 25
            """,
            [ticker],
        ).df()

        if df.empty:
            return df

        df["year"] = pd.to_datetime(df["dt"]).dt.year
        df["quarter"] = pd.to_datetime(df["dt"]).dt.quarter
        df["period"] = df["year"].astype(str) + " Q" + df["quarter"].astype(str)
        return df
    finally:
        con.close()

tickers = load_tickers(DB)
if not tickers:
    st.error("No tickers found in gold.pressure_index. Run `make pipeline`.")
    st.stop()



with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Ticker", tickers, index=0)
    last_n = st.slider("Quarters to show", min_value=4, max_value=24, value=12, step=1)
    threshold = st.slider("Flag threshold (pressure_index)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

df = load_company_quarters(DB, ticker)

if df.empty:
    st.warning(f"No quarterly rows found for {ticker}.")
    st.stop()

# take last N quarters
df = df.sort_values("sort_key").tail(last_n).reset_index(drop=True)

# Choose which quarter to inspect (default: latest)
periods = df["period"].tolist()
selected_period = st.selectbox("Inspect quarter", periods, index=len(periods) - 1)

df = df.sort_values("sort_key").reset_index(drop=True)

row = df[df["period"] == selected_period].iloc[0]
row_idx = df.index[df["period"] == selected_period][0]
prev_row = df.iloc[row_idx - 1] if row_idx - 1 >= 0 else None

def _fmt(v, fmt: str):
    if v is None or pd.isna(v):
        return "NA"
    try:
        return format(float(v), fmt)
    except Exception:
        return "NA"

def _delta(curr, prev, col):
    if prev is None:
        return None
    a = curr.get(col)
    b = prev.get(col)
    if pd.isna(a) or pd.isna(b):
        return None
    try:
        return float(a) - float(b)
    except Exception:
        return None

def _pct(curr, prev, col):
    if prev is None:
        return None
    a = curr.get(col)
    b = prev.get(col)
    if pd.isna(a) or pd.isna(b):
        return None
    try:
        a = float(a); b = float(b)
        return None if b == 0 else (a - b) / b
    except Exception:
        return None


curr = row.to_dict()
prev = prev_row.to_dict() if prev_row is not None else None

st.subheader("Latest snapshot (selected quarter)")
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Period", curr.get("period"))

c2.metric(
    "pressure_index",
    _fmt(curr.get("pressure_index"), ".2f"),
    _fmt(_delta(curr, prev, "pressure_index"), "+.2f") if prev is not None else None,
)

c3.metric(
    "z_pressure_language",
    _fmt(curr.get("z_pressure_language"), ".2f"),
    _fmt(_delta(curr, prev, "z_pressure_language"), "+.2f") if prev is not None else None,
)

c4.metric(
    "z_inventory_to_sales",
    _fmt(curr.get("z_inventory_to_sales"), ".2f"),
    _fmt(_delta(curr, prev, "z_inventory_to_sales"), "+.2f") if prev is not None else None,
)

c5.metric(
    "inventory_to_sales",
    _fmt(curr.get("inventory_to_sales"), ".4f"),
    _fmt(_delta(curr, prev, "inventory_to_sales"), "+.4f") if prev is not None else None,
)

if pd.notna(curr.get("pressure_index")) and float(curr["pressure_index"]) >= threshold:
    st.error(f"Flag: {ticker} {curr['period']} pressure_index ({float(curr['pressure_index']):.2f}) ≥ {threshold:.1f}")



# ---- Trends ----
st.subheader("Trends")

trend = df[["period", "pressure_index"]].set_index("period")
st.line_chart(trend)

components = df[["period", "z_pressure_language", "z_inventory_to_sales"]].set_index("period")
st.line_chart(components)

# ---- What drove selected quarter ----
st.subheader("What drove the selected quarter?")

driver_cols = [
    "kw_inventory", "kw_promotion", "kw_promotional", "kw_markdown",
    "kw_demand", "kw_traffic", "kw_pricing", "kw_guidance",
]

# Pull the keyword counts for the selected quarter
kw = df.loc[row_idx, driver_cols].fillna(0).astype(float)

kw_df = (
    kw.rename(lambda x: x.replace("kw_", ""))
      .to_frame(name="count")          # ensures we have a real "count" column
      .reset_index()
      .rename(columns={"index": "keyword"})
)

total = kw_df["count"].sum()
kw_df["share"] = (kw_df["count"] / total) if total > 0 else 0.0
kw_df = kw_df.sort_values("count", ascending=False).reset_index(drop=True)

left, right = st.columns([1, 1])

with left:
    st.caption("Keyword counts (selected quarter)")
    st.dataframe(kw_df, width="stretch", hide_index=True)

with right:
    st.bar_chart(kw_df.set_index("keyword")[["count"]])


# ---- Detail table ----
st.subheader("Quarterly detail")

show_cols = [
    "year", "quarter", "period",
    "pressure_index", "pressure_language_score",
    "inventory", "revenue", "cogs", "gross_profit", "op_income",
    "inventory_to_sales",
    "inventory_net_qoq_pct", "net_sales_qoq_pct",
    "pressure_qoq_pct", "inventory_to_sales_qoq_pct",
]
detail = df[[c for c in show_cols if c in df.columns]].copy()
st.dataframe(detail.sort_values(["year", "quarter"], ascending=False), width="stretch", hide_index=True)

csv = detail.to_csv(index=False).encode("utf-8")
st.download_button("Download quarterly detail (CSV)", data=csv, file_name=f"{ticker}_deep_dive.csv", mime="text/csv")

# ---- Recent filings (filing-level) ----
st.subheader("Recent filings (filing-level keyword signals)")

fil = load_recent_filings(DB, ticker)
if fil.empty:
    st.info("No filing-level signals found for this ticker.")
else:
    st.dataframe(
        fil[["period", "form", "filing_date", "report_date", "accession"] + driver_cols],
        width="stretch",
        hide_index=True,
    )
st.subheader("Filing Explorer")

if fil.empty:
    st.info("No filings to explore.")
else:
    fil2 = fil.copy()
    fil2["label"] = fil2["period"] + " • " + fil2["form"] + " • " + fil2["accession"]
    fil2["dt"] = pd.to_datetime(fil2["dt"], errors="coerce")
    fil2 = fil2.sort_values("dt", ascending=False)

    label = st.selectbox("Select a filing", fil2["label"].tolist(), index=0)
    accn = fil2.loc[fil2["label"] == label, "accession"].iloc[0]

    ft = load_filing_text(DB, ticker, accn)
    if ft.empty:
        st.error("No filing text found in silver.filing_text for this accession. Re-run `make pipeline`.")
    else:
        meta = ft.iloc[0].to_dict()
        text_blob = meta.get("text") or ""

        st.markdown(f"**SEC URL:** {meta.get('url')}")
        st.link_button("Open on SEC", meta.get("url"))
        st.caption(f"Text length: {int(meta.get('text_len') or 0):,} characters")

        # keyword counts for this filing (from fil row)
        driver_cols = [
            "kw_inventory", "kw_promotion", "kw_promotional", "kw_markdown",
            "kw_demand", "kw_traffic", "kw_pricing", "kw_guidance",
        ]
        r = fil2[fil2["accession"] == accn].iloc[0]
        counts = r[driver_cols].fillna(0).astype(int)

        counts_df = (
            counts.rename(lambda x: x.replace("kw_", ""))
                  .to_frame("count")
                  .reset_index()
                  .rename(columns={"index": "keyword"})
                  .sort_values("count", ascending=False)
        )
        st.dataframe(counts_df, width="stretch", hide_index=True)

        # snippet picker
        default_terms = counts_df.loc[counts_df["count"] > 0, "keyword"].head(3).tolist()
        terms = st.multiselect(
            "Show snippets for keywords",
            options=counts_df["keyword"].tolist(),
            default=default_terms,
        )
        custom = st.text_input("Custom term (optional)", value="")

        terms_to_search = [t for t in terms if t] + ([custom] if custom else [])
        if not terms_to_search:
            st.info("Pick at least one keyword (or type a custom term).")
        else:
            for term in terms_to_search:
                snippets = extract_snippets(text_blob, term, window=180, max_hits=3)
                st.markdown(f"#### `{term}`")
                if not snippets:
                    st.write("No matches found.")
                else:
                    for s in snippets:
                        st.write(s)

        # ---- C3: Top excerpts ----
        st.subheader("Top excerpts")

        top_n = st.slider("How many excerpts?", 5, 25, 10, 1, key="top_excerpt_n")

        ex_terms = [t for t in terms_to_search if t]
        if not ex_terms:
            st.info("Pick at least one keyword (or type a custom term) to generate excerpts.")
        else:
            h = hashlib.sha1(text_blob.encode("utf-8", errors="ignore")).hexdigest()
            excerpts = cached_top_excerpts(h, tuple(ex_terms), int(top_n), text_blob)

            if not excerpts:
                st.info("No excerpts found for those terms.")
            else:
                for i, ex in enumerate(excerpts, start=1):
                    hits_str = ", ".join(
                        f"{k}×{v}" for k, v in sorted(ex["hits"].items(), key=lambda kv: (-kv[1], kv[0]))
                    )
                    st.markdown(f"**#{i}** • **Score:** {ex['score']:.2f} • **Hits:** {hits_str}")

                    box = highlight_terms(ex["text"], ex_terms)
                    st.markdown(
                        f"<div style='padding:0.75rem 0.9rem;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;line-height:1.45'>"
                        f"{box}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


                with st.expander("Preview raw filing text (first 10,000 chars)"):
                    st.text(text_blob[:10000])

                st.download_button(
                    "Download full filing text (TXT)",
                    data=text_blob.encode("utf-8", errors="ignore"),
                    file_name=f"{ticker}_{accn}.txt",
                    mime="text/plain",
                )
