from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

from .edgar_endpoints import filing_primary_doc_url


@dataclass(frozen=True)
class FilingRef:
    cik: int
    accession: str
    form: str
    filing_date: str
    report_date: str | None
    primary_doc: str


def extract_recent_filings(submissions_json: dict, forms: set[str], limit: int) -> list[FilingRef]:
    recent = (submissions_json.get("filings") or {}).get("recent") or {}
    out: list[FilingRef] = []

    for acc, form, fdate, rdate, pdoc in zip(
        recent.get("accessionNumber", []),
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("primaryDocument", []),
    ):
        if form not in forms:
            continue
        out.append(
            FilingRef(
                cik=int(submissions_json["cik"]),
                accession=acc,
                form=form,
                filing_date=fdate,
                report_date=rdate,
                primary_doc=pdoc,
            )
        )
        if len(out) >= limit:
            break

    return out


def filings_to_df(filings: Iterable[FilingRef]) -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in filings])


def _soup_for_content(raw: str) -> BeautifulSoup:
    # Very lightweight detection: XML filings/exhibits often start with '<?xml' or have an <xbrl> root
    head = raw.lstrip()[:200].lower()
    if head.startswith("<?xml") or "<xbrl" in head or "<xml" in head:
        return BeautifulSoup(raw, "xml")  # uses lxml's XML parser (already installed)
    return BeautifulSoup(raw, "lxml")

def parse_primary_document(raw: str) -> str:
    return extract_text_from_raw(raw)

def extract_text_from_raw(raw: str) -> str:
    soup = _soup_for_content(raw)

    # If HTML, remove script/style/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    # collapse extra whitespace
    return " ".join(text.split())


def compute_keyword_counts(text: str, keywords: list[str]) -> dict[str, int]:
    lower = text.lower()
    return {f"kw_{k.lower()}": lower.count(k.lower()) for k in keywords}


def primary_doc_url(f: FilingRef) -> str:
    # Prefer HTML if primary doc is XML
    pdoc = f.primary_doc
    if pdoc.lower().endswith(".xml"):
        # common fallback names used by filings; you can expand this list
        for candidate in ["d10k.htm", "d10q.htm", "form10k.htm", "form10q.htm", "10k.htm", "10q.htm"]:
            # try candidate only if your pipeline supports trying alternates
            pass
    return filing_primary_doc_url(f.cik, f.accession, pdoc)

def html_to_text(html: str) -> str:
    """
    Backwards-compatible name expected by build.py.
    Accepts HTML or XML-ish content and returns normalized text.
    """
    return extract_text_from_raw(html)
