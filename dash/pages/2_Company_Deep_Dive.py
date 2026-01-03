from __future__ import annotations
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from pathlib import Path
from difflib import SequenceMatcher
import duckdb
import io
import zipfile
import pandas as pd
import streamlit as st
import re
import html
import hashlib
import json

def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _txt_bytes(s: str) -> bytes:
    return (s or "").encode("utf-8", errors="ignore")

def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")

def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    """
    files: {"path/in/zip.ext": b"..."}
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            if data is None:
                continue
            z.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()

def download_zip_button(label: str, zip_filename: str, files: dict[str, bytes], key: str):
    zbytes = make_zip_bytes(files)
    st.download_button(
        label=label,
        data=zbytes,
        file_name=zip_filename,
        mime="application/zip",
        key=key,
    )

def _hits_to_str(hits: dict) -> str:
    if not hits:
        return ""
    return ", ".join(
        f"{k}×{v}" for k, v in sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    )

def excerpts_to_df(excerpts: list[dict], scope: str) -> pd.DataFrame:
    """
    Convert excerpts list-of-dicts into a flat dataframe for export.
    scope is just a label like 'quarter' or 'filing' to help identify exports later.
    """
    rows = []
    for i, ex in enumerate(excerpts or [], start=1):
        hits = ex.get("hits") or {}
        rows.append(
            {
                "scope": scope,
                "rank": i,
                "score": float(ex.get("score") or 0.0),
                "hits": _hits_to_str(hits),
                "hits_json": json.dumps(hits, ensure_ascii=False),
                "form": ex.get("form"),
                "dt": ex.get("dt"),
                "accession": ex.get("accession"),
                "url": ex.get("url"),
                "text": ex.get("text") or "",
            }
        )
    return pd.DataFrame(rows)

def build_excerpt_summary_md(
    excerpts: list[dict],
    title: str,
    top_terms: int = 5,
    top_excerpts: int = 3,
) -> str:
    """
    Simple summary:
      - Top terms by total hits across selected excerpts
      - A few highest-scoring excerpts with metadata
    """
    excerpts = excerpts or []
    if not excerpts:
        return f"### {title}\n\n_No excerpts found._"

    # aggregate hits
    agg: dict[str, int] = {}
    for ex in excerpts:
        hits = ex.get("hits") or {}
        for k, v in hits.items():
            try:
                agg[k] = agg.get(k, 0) + int(v)
            except Exception:
                agg[k] = agg.get(k, 0) + 1

    top_terms_list = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[:top_terms]

    lines = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append("**Top themes (by hits):**")
    if top_terms_list:
        lines.extend([f"- {k}: {v}" for k, v in top_terms_list])
    else:
        lines.append("- (no term hits aggregated)")
    lines.append("")
    lines.append("**Best excerpts:**")

    # take best N by score
    best = sorted(excerpts, key=lambda ex: float(ex.get("score") or 0.0), reverse=True)[:top_excerpts]
    for i, ex in enumerate(best, start=1):
        meta = " • ".join(
            [str(x) for x in [ex.get("form"), ex.get("dt"), ex.get("accession")] if x]
        )
        hits_str = _hits_to_str(ex.get("hits") or {})
        txt = (ex.get("text") or "").strip().replace("\n", " ")
        # keep it paste-friendly
        if len(txt) > 360:
            txt = txt[:360].rstrip() + "…"
        lines.append(f"{i}. ({hits_str}) {meta}")
        lines.append(f"   - {txt}")

    return "\n".join(lines)

def download_csv_button(df: pd.DataFrame, label: str, filename: str, key: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv", key=key)


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

# ---- C6: Save exports to disk (helpers) -----------------------------------

EXPORTS_DIR = ROOT / "data" / "exports"  # local-only; add to .gitignore if desired

def _safe(s: str) -> str:
    """Safe filename-ish token."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s or "").strip())
    return s.strip("_") or "NA"

def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8", errors="ignore")

def _save_bundle_to_disk(
    out_dir: Path,
    zip_name: str,
    zip_bytes: bytes,
    loose_files: dict[str, bytes] | None = None,
) -> dict[str, str]:
    """
    Writes:
      - out_dir/zip_name
      - out_dir/<loose_file_keys...> (optional)
    Returns paths for UI display.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / zip_name
    _write_bytes(zip_path, zip_bytes)

    written = {"folder": str(out_dir), "zip": str(zip_path)}

    if loose_files:
        for rel_name, b in loose_files.items():
            if b is None:
                continue
            rel_name = rel_name.lstrip("/").replace("\\", "/")
            file_path = out_dir / rel_name
            _write_bytes(file_path, b)
            written[rel_name] = str(file_path)

    return written

def _show_saved_paths(written: dict[str, str], title: str = "Saved to disk"):
    st.success(title)
    st.code(written.get("folder", ""), language=None)
    # Show key files
    for k in ["zip", "summary.md", "run_context.json"]:
        if k in written:
            st.caption(f"{k}:")
            st.code(written[k], language=None)

# ---- End C6 helpers --------------------------------------------------------

 #---- Start of Drop in PDF helpers --------------------------------------------------------

def _pdf_bytes_quarter_brief(
    *,
    ticker: str,
    period: str,
    threshold: float,
    snapshot: dict,
    kw_df: pd.DataFrame,
    excerpts_df: pd.DataFrame,
    summary_md: str,
) -> bytes:
    """
    Build a simple PDF executive brief for the selected quarter.
    Returns PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=f"{ticker} {period} - Executive Brief",
        author="edgar-retail-etl",
    )

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6)
    BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9, leading=12)
    MONO = ParagraphStyle("MONO", parent=styles["BodyText"], fontName="Courier", fontSize=8, leading=10)

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.45 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    story = []

    # Title
    story.append(Paragraph(f"{ticker} - {period} Executive Brief", H1))
    story.append(Paragraph("Source: SEC EDGAR filings (DuckDB gold tables).", BODY))
    story.append(Spacer(1, 8))

    # Flag line
    try:
        pidx = float(snapshot.get("pressure_index")) if snapshot.get("pressure_index") is not None else None
    except Exception:
        pidx = None

    flag_txt = "Flag: NO"
    if pidx is not None and pidx >= float(threshold):
        flag_txt = f"Flag: YES (pressure_index {pidx:.2f} >= {float(threshold):.1f})"
    elif pidx is not None:
        flag_txt = f"Flag: NO (pressure_index {pidx:.2f} < {float(threshold):.1f})"

    story.append(Paragraph(flag_txt, BODY))
    story.append(Spacer(1, 10))

    # Snapshot table
    story.append(Paragraph("Snapshot", H2))
    snap_rows = [
        ["Metric", "Value"],
        ["pressure_index", _safe_fmt(snapshot.get("pressure_index"), ".2f")],
        ["z_pressure_language", _safe_fmt(snapshot.get("z_pressure_language"), ".2f")],
        ["z_inventory_to_sales", _safe_fmt(snapshot.get("z_inventory_to_sales"), ".2f")],
        ["inventory_to_sales", _safe_fmt(snapshot.get("inventory_to_sales"), ".4f")],
        ["pressure_language_score", _safe_fmt(snapshot.get("pressure_language_score"), ".0f")],
        ["inventory", _safe_fmt(snapshot.get("inventory"), ".0f")],
        ["revenue", _safe_fmt(snapshot.get("revenue"), ".0f")],
    ]
    t = Table(snap_rows, colWidths=[2.2 * inch, 3.6 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F9FAFB")]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    # Keyword drivers (top 10)
    story.append(Paragraph("Keyword drivers (top 10)", H2))
    kw_small = kw_df.copy()
    if "keyword" in kw_small.columns and "count" in kw_small.columns:
        kw_small = kw_small.sort_values("count", ascending=False).head(10)
        kw_rows = [["keyword", "count", "share"]]
        for _, r in kw_small.iterrows():
            kw_rows.append(
                [
                    str(r.get("keyword") or ""),
                    str(int(r.get("count") or 0)),
                    f"{float(r.get('share') or 0.0):.2%}",
                ]
            )
        kwt = Table(kw_rows, colWidths=[2.2 * inch, 1.0 * inch, 1.0 * inch])
        kwt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D1D5DB")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ]
            )
        )
        story.append(kwt)
    else:
        story.append(Paragraph("Keyword table not available.", BODY))
    story.append(Spacer(1, 12))

    # Summary + best excerpts
    story.append(Paragraph("Summary", H2))
    for line in (summary_md or "").splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        # keep it simple: render as plain paragraphs
        story.append(Paragraph(_escape_pdf(line), BODY))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Top excerpts (ranked)", H2))
    if excerpts_df is None or excerpts_df.empty:
        story.append(Paragraph("No excerpts available.", BODY))
    else:
        # Keep it readable: show top 6 in the PDF
        topk = excerpts_df.sort_values("rank").head(6).copy()
        for _, r in topk.iterrows():
            meta = " - ".join([str(x) for x in [r.get("form"), r.get("dt"), r.get("accession")] if x])
            story.append(Paragraph(f"Rank {int(r.get('rank') or 0)} | Score {float(r.get('score') or 0):.2f}", BODY))
            if meta:
                story.append(Paragraph(_escape_pdf(meta), MONO))
            url = str(r.get("url") or "").strip()
            if url:
                story.append(Paragraph(_escape_pdf(url), MONO))

            txt = str(r.get("text") or "").replace("\n", " ").strip()
            if len(txt) > 900:
                txt = txt[:900].rstrip() + "..."
            story.append(Paragraph(_escape_pdf(txt), BODY))
            story.append(Spacer(1, 8))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.getvalue()


def _escape_pdf(s: str) -> str:
    # ReportLab Paragraph is HTML-ish. Escape angle brackets etc.
    s = (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s


def _safe_fmt(v, fmt: str) -> str:
    try:
        if v is None or pd.isna(v):
            return "NA"
        return format(float(v), fmt)
    except Exception:
        return "NA"
# ---- End of Drop in PDF helpers --------------------------------------------------------

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

@st.cache_data(show_spinner=False)
def load_filing_texts_for_period(
    db_path: Path,
    ticker: str,
    year: int,
    quarter: int,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Pull filing text rows for a ticker in a given year/quarter.
    Uses silver.filing_text.dt to compute year/quarter.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
              ticker,
              accession,
              form,
              filing_date,
              report_date,
              dt,
              url,
              text,
              text_len
            FROM silver.filing_text
            WHERE ticker = ?
              AND dt IS NOT NULL
              AND EXTRACT(year FROM CAST(dt AS DATE)) = ?
              AND EXTRACT(quarter FROM CAST(dt AS DATE)) = ?
            ORDER BY CAST(dt AS DATE) DESC
            LIMIT ?
            """,
            [ticker, int(year), int(quarter), int(limit)],
        ).df()
    finally:
        con.close()




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

def top_excerpts_multi(
    filings_df: pd.DataFrame,
    terms: list[str],
    top_n: int = 10,
    per_filing_cap: int = 6,
) -> list[dict]:
    """
    Generate top excerpts across multiple filings in a quarter.

    Strategy:
      - run top_excerpts per filing (cap per filing)
      - combine + near-duplicate suppress globally (re-uses _is_too_similar)
      - keep best top_n overall
    """
    terms = [t.strip() for t in terms if t and t.strip()]
    if filings_df is None or filings_df.empty or not terms:
        return []

    combined: list[dict] = []

    for _, r in filings_df.iterrows():
        txt = r.get("text") or ""
        if not txt:
            continue

        exs = top_excerpts(txt, terms, top_n=per_filing_cap)
        for ex in exs:
            combined.append(
                {
                    "score": ex["score"],
                    "hits": ex["hits"],
                    "text": ex["text"],
                    "accession": r.get("accession"),
                    "form": r.get("form"),
                    "dt": r.get("dt"),
                    "url": r.get("url"),
                }
            )

    combined.sort(key=lambda x: x["score"], reverse=True)

    picked: list[dict] = []
    for ex in combined:
        if any(_is_too_similar(ex["text"], p["text"]) for p in picked):
            continue
        picked.append(ex)
        if len(picked) >= top_n:
            break

    return picked

@st.cache_data(show_spinner=False)
def cached_quarter_excerpts(
    ticker: str,
    year: int,
    quarter: int,
    terms_tuple: tuple[str, ...],
    top_n: int,
    filings_limit: int,
    per_filing_cap: int,
    db_path_str: str,
):
    dfq = load_filing_texts_for_period(Path(db_path_str), ticker, year, quarter, limit=filings_limit)
    return top_excerpts_multi(dfq, list(terms_tuple), top_n=top_n, per_filing_cap=per_filing_cap)


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
kw = (
    df.loc[row_idx, driver_cols]
      .infer_objects(copy=False)
      .fillna(0)
      .astype(float)
)


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

# ---- C4: Quarter excerpts (across filings in selected quarter) ----
st.subheader("Quarter excerpts (across filings)")

q_year = int(row["year"])
q_quarter = int(row["quarter"])

# controls
cA, cB, cC = st.columns([2, 1, 1])
with cA:
    # Use quarter keywords as options; default: top 3 nonzero
    quarter_keyword_options = kw_df["keyword"].tolist()
    default_q_terms = kw_df.loc[kw_df["count"] > 0, "keyword"].head(3).tolist()
    q_terms = st.multiselect(
        "Terms to rank excerpts by",
        options=quarter_keyword_options,
        default=default_q_terms,
        key="q_excerpt_terms",
    )
    q_custom = st.text_input("Custom term (optional)", value="", key="q_excerpt_custom")
with cB:
    filings_limit = st.slider("Filings to scan", 1, 25, 8, 1, key="q_excerpt_filings_limit")
with cC:
    q_top_n = st.slider("Excerpts to show", 3, 20, 8, 1, key="q_excerpt_top_n")

q_terms_to_use = [t for t in q_terms if t] + ([q_custom] if q_custom else [])

if not q_terms_to_use:
    st.info("Pick at least one term (or add a custom term) to generate quarter excerpts.")
else:
    # You can use cached_quarter_excerpts (recommended) or call directly.
    excerpts = cached_quarter_excerpts(
        ticker=ticker,
        year=q_year,
        quarter=q_quarter,
        terms_tuple=tuple(q_terms_to_use),
        top_n=int(q_top_n),
        filings_limit=int(filings_limit),
        per_filing_cap=6,
        db_path_str=str(DB),
    )

    # ---- C7: Executive quarter summary ----------------------------------------

    def _pp(x: float | None) -> str:
        if x is None or pd.isna(x):
            return "NA"
        try:
            return f"{float(x)*100:+.1f}pp"
        except Exception:
            return "NA"

    def _pct_str(x: float | None) -> str:
        if x is None or pd.isna(x):
            return "NA"
        try:
            return f"{float(x)*100:+.1f}%"
        except Exception:
            return "NA"

    def _num(x, fmt=".2f") -> str:
        if x is None or pd.isna(x):
            return "NA"
        try:
            return format(float(x), fmt)
        except Exception:
            return "NA"

    def _get_prev_kw_df(df_all: pd.DataFrame, idx: int, driver_cols: list[str]) -> pd.DataFrame | None:
        """Build previous-quarter keyword df in same format as kw_df (keyword,count,share)."""
        if idx <= 0:
            return None
        prev_kw = (
            df_all.loc[idx - 1, driver_cols]
                .infer_objects(copy=False)
                .fillna(0)
                .astype(float)
        )
        prev_kw_df = (
            prev_kw.rename(lambda x: x.replace("kw_", ""))
                .to_frame(name="count")
                .reset_index()
                .rename(columns={"index": "keyword"})
        )
        total_prev = prev_kw_df["count"].sum()
        prev_kw_df["share"] = (prev_kw_df["count"] / total_prev) if total_prev > 0 else 0.0
        return prev_kw_df

    def _keyword_mix_movers(kw_curr_df: pd.DataFrame, kw_prev_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Return top share movers (pp) for the quarter keyword mix."""
        if kw_prev_df is None or kw_prev_df.empty:
            out = kw_curr_df.copy()
            out["prev_share"] = 0.0
            out["share_pp"] = out["share"] - out["prev_share"]
            return out.sort_values(["count", "keyword"], ascending=[False, True]).head(top_k)

        a = kw_curr_df[["keyword", "count", "share"]].copy()
        b = kw_prev_df[["keyword", "share"]].copy().rename(columns={"share": "prev_share"})
        m = a.merge(b, on="keyword", how="left")
        m["prev_share"] = m["prev_share"].fillna(0.0)
        m["share_pp"] = m["share"] - m["prev_share"]
        # “movers” = biggest absolute share change, break ties by higher count
        m["abs_pp"] = m["share_pp"].abs()
        return m.sort_values(["abs_pp", "count", "keyword"], ascending=[False, False, True]).head(top_k)

    def build_exec_summary_md(
        ticker: str,
        selected_period: str,
        curr: dict,
        prev: dict | None,
        kw_curr_df: pd.DataFrame,
        kw_prev_df: pd.DataFrame | None,
        movers_df: pd.DataFrame,
        excerpts: list[dict],
        terms_used: list[str],
        threshold: float,
        evidence_n: int = 3,
    ) -> str:
        lines: list[str] = []

        # Header
        lines.append(f"## {ticker} — Executive Summary ({selected_period})")
        lines.append("")

        # Flag line
        pi = curr.get("pressure_index")
        flag = ""
        try:
            if pi is not None and not pd.isna(pi) and float(pi) >= float(threshold):
                flag = f"⚠️ **Flagged:** pressure_index {float(pi):.2f} ≥ {threshold:.1f}"
        except Exception:
            flag = ""
        if flag:
            lines.append(flag)
            lines.append("")

        # Metrics snapshot
        lines.append("### 1) Pressure snapshot (QoQ)")
        if prev is None:
            lines.append(f"- pressure_index: {_num(curr.get('pressure_index'), '.2f')} (no prior quarter to compare)")
            lines.append(f"- z_pressure_language: {_num(curr.get('z_pressure_language'), '.2f')}")
            lines.append(f"- z_inventory_to_sales: {_num(curr.get('z_inventory_to_sales'), '.2f')}")
            lines.append(f"- inventory_to_sales: {_num(curr.get('inventory_to_sales'), '.4f')}")
            lines.append(f"- pressure_language_score: {_num(curr.get('pressure_language_score'), '.0f')}")
        else:
            lines.append(
                f"- pressure_index: {_num(curr.get('pressure_index'), '.2f')} "
                f"({_num(float(curr.get('pressure_index')) - float(prev.get('pressure_index')), '+.2f') if curr.get('pressure_index') is not None and prev.get('pressure_index') is not None else 'NA'} QoQ)"
            )
            lines.append(
                f"- z_pressure_language: {_num(curr.get('z_pressure_language'), '.2f')} "
                f"({_num(float(curr.get('z_pressure_language')) - float(prev.get('z_pressure_language')), '+.2f') if curr.get('z_pressure_language') is not None and prev.get('z_pressure_language') is not None else 'NA'} QoQ)"
            )
            lines.append(
                f"- z_inventory_to_sales: {_num(curr.get('z_inventory_to_sales'), '.2f')} "
                f"({_num(float(curr.get('z_inventory_to_sales')) - float(prev.get('z_inventory_to_sales')), '+.2f') if curr.get('z_inventory_to_sales') is not None and prev.get('z_inventory_to_sales') is not None else 'NA'} QoQ)"
            )
            # ratio + already-computed pct features if present
            lines.append(
                f"- inventory_to_sales: {_num(curr.get('inventory_to_sales'), '.4f')} "
                f"({ _pct_str(curr.get('inventory_to_sales_qoq_pct')) } QoQ)"
            )
            lines.append(
                f"- pressure_language_score: {_num(curr.get('pressure_language_score'), '.0f')} "
                f"({ _pct_str(curr.get('pressure_qoq_pct')) } QoQ)"
            )
            lines.append(
                f"- inventory (net): {_num(curr.get('inventory'), '.0f')} ({ _pct_str(curr.get('inventory_net_qoq_pct')) } QoQ) "
                f"| revenue: {_num(curr.get('revenue'), '.0f')} ({ _pct_str(curr.get('net_sales_qoq_pct')) } QoQ)"
            )
        lines.append("")

        # Keyword mix
        lines.append("### 2) Language mix (keyword share movers)")
        if movers_df is None or movers_df.empty:
            lines.append("- No keyword mix data available.")
        else:
            for _, r in movers_df.iterrows():
                lines.append(
                    f"- {r['keyword']}: {int(r['count']) if not pd.isna(r['count']) else 0} hits | "
                    f"share {_pct_str(r['share'])} (prev {_pct_str(r.get('prev_share'))}) | Δ {_pp(r.get('share_pp'))}"
                )
        lines.append("")

        # Evidence
        lines.append("### 3) Evidence (top excerpts)")
        lines.append(f"- Terms used: {', '.join(terms_used) if terms_used else '(none)'}")
        if not excerpts:
            lines.append("- No excerpts found for selected terms.")
        else:
            best = sorted(excerpts, key=lambda ex: float(ex.get("score") or 0.0), reverse=True)[:evidence_n]
            for i, ex in enumerate(best, start=1):
                meta = " • ".join([str(x) for x in [ex.get("form"), ex.get("dt"), ex.get("accession")] if x])
                hits_str = _hits_to_str(ex.get("hits") or {})
                txt = (ex.get("text") or "").strip().replace("\n", " ")
                if len(txt) > 420:
                    txt = txt[:420].rstrip() + "…"
                lines.append(f"{i}. ({hits_str}) {meta}")
                lines.append(f"   - {txt}")
                if ex.get("url"):
                    lines.append(f"   - Source: {ex['url']}")
        lines.append("")

        # One-liner takeaway
        lines.append("### 4) Takeaway")
        # lightweight heuristic: prioritize pressure_index direction if available
        if prev is not None and curr.get("pressure_index") is not None and prev.get("pressure_index") is not None:
            try:
                d = float(curr["pressure_index"]) - float(prev["pressure_index"])
                if d > 0.25:
                    lines.append(f"- Pressure **increased** QoQ (Δ {d:+.2f}). Language + inventory signals are trending hotter this quarter.")
                elif d < -0.25:
                    lines.append(f"- Pressure **decreased** QoQ (Δ {d:+.2f}). Signals cooled relative to last quarter.")
                else:
                    lines.append(f"- Pressure is **roughly flat** QoQ (Δ {d:+.2f}). Mix shifts may be more informative than level changes.")
            except Exception:
                lines.append("- Pressure changed QoQ; see metrics above for details.")
        else:
            lines.append("- Review the metrics + evidence above to determine whether pressure is building or easing.")
        lines.append("")

        return "\n".join(lines)

    # ---- UI integration (place inside your quarter section after excerpts exist) ----
    st.divider()
    st.subheader("Executive summary (Quarter)")

    # Build previous keyword mix + movers
    kw_prev_df = _get_prev_kw_df(df, row_idx, driver_cols)
    movers_df = _keyword_mix_movers(kw_df, kw_prev_df, top_k=6)

    # Build the executive summary markdown
    exec_md = build_exec_summary_md(
        ticker=ticker,
        selected_period=selected_period,
        curr=curr,
        prev=prev,
        kw_curr_df=kw_df,
        kw_prev_df=kw_prev_df,
        movers_df=movers_df,
        excerpts=excerpts,
        terms_used=q_terms_to_use,
        threshold=float(threshold),
        evidence_n=3,
    )

    # Show + export
    st.markdown(exec_md)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download executive summary (MD)",
            data=_txt_bytes(exec_md),
            file_name=f"{ticker}_{q_year}Q{q_quarter}_exec_summary.md",
            mime="text/markdown",
            key="dl_exec_summary_md",
        )
    with c2:
        st.download_button(
            "Download executive summary (TXT)",
            data=_txt_bytes(exec_md),
            file_name=f"{ticker}_{q_year}Q{q_quarter}_exec_summary.txt",
            mime="text/plain",
            key="dl_exec_summary_txt",
        )

    with st.expander("Copy/paste executive summary"):
        st.text_area("Executive summary", value=exec_md, height=260, key="exec_summary_textarea")

    st.caption("Keyword movers (debug view)")
    st.dataframe(movers_df.drop(columns=["abs_pp"], errors="ignore"), width="stretch", hide_index=True)

    # ---- End C7 ----------------------------------------------------------------



    # ---- C5: Export + Summary (Quarter) ----
    st.divider()
    st.subheader("Quarter exports + summary")

    q_df = excerpts_to_df(excerpts, scope="quarter")

    c1, c2 = st.columns([1, 1])
    with c1:
        download_csv_button(
            q_df,
            label="Download quarter excerpts (CSV)",
            filename=f"{ticker}_{q_year}Q{q_quarter}_quarter_excerpts.csv",
            key="dl_quarter_excerpts_csv",
        )
    with c2:
        # optional: a "light" export without full text (sometimes nicer)
        light = q_df.drop(columns=["text"], errors="ignore")
        download_csv_button(
            light,
            label="Download quarter excerpts (CSV, no text)",
            filename=f"{ticker}_{q_year}Q{q_quarter}_quarter_excerpts_notext.csv",
            key="dl_quarter_excerpts_csv_notext",
        )

    summary_md = build_excerpt_summary_md(
        excerpts,
        title=f"{ticker} {q_year} Q{q_quarter} — Quarter Excerpts Summary",
        top_terms=6,
        top_excerpts=3,
    )
    st.markdown(summary_md)
       
    # Build PDF
    snapshot_ctx = {
        "pressure_index": curr.get("pressure_index"),
        "z_pressure_language": curr.get("z_pressure_language"),
        "z_inventory_to_sales": curr.get("z_inventory_to_sales"),
        "inventory_to_sales": curr.get("inventory_to_sales"),
        "pressure_language_score": curr.get("pressure_language_score"),
        "inventory": curr.get("inventory"),
        "revenue": curr.get("revenue"),
    }

    pdf_bytes = _pdf_bytes_quarter_brief(
        ticker=ticker,
        period=selected_period,
        threshold=float(threshold),
        snapshot=snapshot_ctx,
        kw_df=kw_df,
        excerpts_df=q_df,   # includes text + url + meta
        summary_md=summary_md,
    )



    st.download_button(
        "Download quarter brief (PDF)",
        data=pdf_bytes,
        file_name=f"{ticker}_{q_year}Q{q_quarter}_quarter_brief.pdf",
        mime="application/pdf",
        key="dl_quarter_brief_pdf",
    )


    with st.expander("Copy/paste summary (plain text)"):
        st.text_area(
            "Summary",
            value=summary_md,
            height=220,
            key="quarter_summary_text",
        )
    
    # ---- C5/C6: ZIP bundle (Quarter) ------------------------------------------
    st.subheader("Download quarter bundle (ZIP)")

    # Build quarter_detail_df here (so it exists even before the "Quarterly detail" section)
    show_cols_for_bundle = [
        "year", "quarter", "period",
        "pressure_index", "pressure_language_score",
        "inventory", "revenue", "cogs", "gross_profit", "op_income",
        "inventory_to_sales",
        "inventory_net_qoq_pct", "net_sales_qoq_pct",
        "pressure_qoq_pct", "inventory_to_sales_qoq_pct",
    ]
    quarter_detail_df = df[[c for c in show_cols_for_bundle if c in df.columns]].copy()
    quarter_detail_df = quarter_detail_df.sort_values(["year", "quarter"], ascending=False)

    ctx = {
        "scope": "quarter",
        "ticker": ticker,
        "selected_period": selected_period,
        "year": int(q_year),
        "quarter": int(q_quarter),
        "terms": q_terms_to_use,
        "filings_limit": int(filings_limit),
        "top_n": int(q_top_n),
    }

    # build zip payload (files inside the zip)
    zip_files = {
        "quarter/quarter_excerpts.csv": _csv_bytes(q_df),
        "quarter/quarter_excerpts_notext.csv": _csv_bytes(light),
        "quarter/quarter_keyword_counts.csv": _csv_bytes(kw_df),
        "quarter/quarter_detail.csv": _csv_bytes(quarter_detail_df),
        "quarter/summary.md": _txt_bytes(summary_md),
        "quarter/run_context.json": _json_bytes(ctx),
    }
    zip_files["quarter/quarter_brief.pdf"] = pdf_bytes
    zip_files["quarter/exec_summary.md"] = _txt_bytes(exec_md)

    # bytes for the downloadable zip
    quarter_zip_bytes = make_zip_bytes(zip_files)

    download_zip_button(
        label="Download quarter bundle (ZIP)",
        zip_filename=f"{ticker}_{q_year}Q{q_quarter}_quarter_bundle.zip",
        files=zip_files,
        key="dl_quarter_bundle_zip",
    )

    # ---- C6: optionally save to disk -------------------------------------------
    save_quarter = st.checkbox("Also save quarter bundle to disk", value=False, key="save_quarter_bundle_disk")

    if save_quarter:
        out_dir = EXPORTS_DIR / _safe(ticker) / f"{int(q_year)}Q{int(q_quarter)}"
        zip_name = f"{_safe(ticker)}_{int(q_year)}Q{int(q_quarter)}_quarter_bundle.zip"

        # also write "loose" (un-zipped) copies for convenience
        loose = {
            "quarter_excerpts.csv": _csv_bytes(q_df),
            "quarter_excerpts_notext.csv": _csv_bytes(light),
            "quarter_keyword_counts.csv": _csv_bytes(kw_df),
            "quarter_detail.csv": _csv_bytes(quarter_detail_df),
            "summary.md": _txt_bytes(summary_md),
            "run_context.json": _json_bytes(ctx),
            "quarter_brief.pdf": pdf_bytes,
            "exec_summary.md": _txt_bytes(exec_md),
        }

        try:
            written = _save_bundle_to_disk(out_dir, zip_name, quarter_zip_bytes, loose_files=loose)
            # add a couple “well-known” keys for display
            written["summary.md"] = str(out_dir / "summary.md")
            written["run_context.json"] = str(out_dir / "run_context.json")
            _show_saved_paths(written, title="Quarter bundle saved to disk")
        except Exception as e:
            st.error(f"Failed to save quarter bundle to disk: {e}")
    # ---------------------------------------------------------------------------


    if not excerpts:
        st.info("No excerpts found for those terms in this quarter.")
    else:
        for i, ex in enumerate(excerpts, start=1):
            hits_str = ", ".join(
                f"{k}×{v}" for k, v in sorted(ex["hits"].items(), key=lambda kv: (-kv[1], kv[0]))
            )
            meta = f"{ex.get('form') or ''} • {ex.get('dt') or ''} • {ex.get('accession') or ''}".strip(" •")

            st.markdown(f"**#{i}** • **Score:** {ex['score']:.2f} • **Hits:** {hits_str}")
            st.caption(meta)

            if ex.get("url"):
                st.markdown(f"[Open filing on SEC]({ex['url']})")


            box = highlight_terms(ex["text"], q_terms_to_use)
            st.markdown(
                f"<div style='padding:0.75rem 0.9rem;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;line-height:1.45'>"
                f"{box}"
                f"</div>",
                unsafe_allow_html=True,
            )


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
        if meta.get("url"):
            st.markdown(f"[Open on SEC]({meta['url']})")

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
                                # ---- C5: Export + Summary (Single filing) ----
                st.divider()
                st.subheader("Filing exports + summary")

                # build export df (add metadata so exports are useful)
                filing_excerpts = []
                for ex in excerpts:
                    filing_excerpts.append(
                        {
                            **ex,
                            "accession": accn,
                            "form": meta.get("form"),
                            "dt": meta.get("dt"),
                            "url": meta.get("url"),
                        }
                    )

                f_df = excerpts_to_df(filing_excerpts, scope="filing")

                download_csv_button(
                    f_df,
                    label="Download filing excerpts (CSV)",
                    filename=f"{ticker}_{accn}_filing_excerpts.csv",
                    key="dl_filing_excerpts_csv",
                )

                f_summary_md = build_excerpt_summary_md(
                    filing_excerpts,
                    title=f"{ticker} {accn} — Filing Excerpts Summary",
                    top_terms=6,
                    top_excerpts=3,
                )
                st.markdown(f_summary_md)

                with st.expander("Copy/paste summary (plain text)"):
                    st.text_area(
                        "Summary",
                        value=f_summary_md,
                        height=220,
                        key="filing_summary_text",
                    )
                                # ---- C5: ZIP bundle (Filing) ----
                st.subheader("Download filing bundle (ZIP)")

                filing_ctx = {
                    "ticker": ticker,
                    "accession": accn,
                    "form": meta.get("form"),
                    "dt": str(meta.get("dt") or ""),
                    "sec_url": meta.get("url"),
                    "terms": ex_terms,
                    "top_n": int(top_n),
                }

                filing_zip_files = {
                    "filing/filing_excerpts.csv": _csv_bytes(f_df),
                    "filing/keyword_counts.csv": _csv_bytes(counts_df),
                    "filing/summary.md": _txt_bytes(f_summary_md),
                    "filing/run_context.json": _json_bytes(filing_ctx),
                    "filing/filing_text.txt": _txt_bytes(text_blob),
                }

                download_zip_button(
                    label="Download filing bundle (ZIP)",
                    zip_filename=f"{ticker}_{accn}_filing_bundle.zip",
                    files=filing_zip_files,
                    key="dl_filing_bundle_zip",
                )

                # ---- C6: optionally save filing bundle to disk -----------------------------
                save_filing = st.checkbox(
                    "Also save filing bundle to disk",
                    value=False,
                    key="save_filing_bundle_disk",
                )

                if save_filing:
                    out_dir = EXPORTS_DIR / _safe(ticker) / "filings" / _safe(accn)
                    zip_name = f"{_safe(ticker)}_{_safe(accn)}_filing_bundle.zip"

                    filing_zip_bytes = make_zip_bytes(filing_zip_files)

                    loose = {
                        "filing_excerpts.csv": _csv_bytes(f_df),
                        "keyword_counts.csv": _csv_bytes(counts_df),
                        "summary.md": _txt_bytes(f_summary_md),
                        "run_context.json": _json_bytes(filing_ctx),
                        "filing_text.txt": _txt_bytes(text_blob),
                    }

                    try:
                        written = _save_bundle_to_disk(out_dir, zip_name, filing_zip_bytes, loose_files=loose)
                        written["summary.md"] = str(out_dir / "summary.md")
                        written["run_context.json"] = str(out_dir / "run_context.json")
                        _show_saved_paths(written, title="Filing bundle saved to disk")
                    except Exception as e:
                        st.error(f"Failed to save filing bundle to disk: {e}")

                # ---- Always-available raw text tools ---------------------------------------
                with st.expander("Preview raw filing text (first 10,000 chars)"):
                    st.text(text_blob[:10000])

                st.download_button(
                    "Download full filing text (TXT)",
                    data=text_blob.encode("utf-8", errors="ignore"),
                    file_name=f"{ticker}_{accn}.txt",
                    mime="text/plain",
                )
