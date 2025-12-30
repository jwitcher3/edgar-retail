from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SecSettings:
    user_agent: str
    max_requests_per_second: float = 5.0
    timeout_seconds: int = 30


@dataclass(frozen=True)
class ProjectSettings:
    tickers: list[str]
    forms: list[str]
    filings_per_company: int = 8


@dataclass(frozen=True)
class SignalsSettings:
    keywords: list[str]


@dataclass(frozen=True)
class XbrlSettings:
    tags: list[str]


@dataclass(frozen=True)
class Settings:
    sec: SecSettings
    project: ProjectSettings
    signals: SignalsSettings
    xbrl: XbrlSettings
    root_dir: Path


def load_settings(config_path: str | Path) -> Settings:
    config_path = Path(config_path)
    root_dir = config_path.parent.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    sec = raw.get("sec", {})
    proj = raw.get("project", {})
    sig = raw.get("signals", {})
    xbrl = raw.get("xbrl", {})

    user_agent = str(sec.get("user_agent", "")).strip()
    if not user_agent or "@" not in user_agent:
        raise ValueError("sec.user_agent must include contact info (e.g., an email).")

    return Settings(
        sec=SecSettings(
            user_agent=user_agent,
            max_requests_per_second=float(sec.get("max_requests_per_second", 5)),
            timeout_seconds=int(sec.get("timeout_seconds", 30)),
        ),
        project=ProjectSettings(
            tickers=list(proj.get("tickers", [])),
            forms=list(proj.get("forms", ["10-K", "10-Q"])),
            filings_per_company=int(proj.get("filings_per_company", 8)),
        ),
        signals=SignalsSettings(keywords=list(sig.get("keywords", []))),
        xbrl=XbrlSettings(tags=list(xbrl.get("tags", []))),
        root_dir=root_dir,
    )
