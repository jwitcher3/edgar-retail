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


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def compute_keyword_counts(text: str, keywords: list[str]) -> dict[str, int]:
    lower = text.lower()
    return {f"kw_{k.lower()}": lower.count(k.lower()) for k in keywords}


def primary_doc_url(f: FilingRef) -> str:
    return filing_primary_doc_url(f.cik, f.accession, f.primary_doc)
