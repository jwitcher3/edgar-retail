from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class SecHttpError(RuntimeError):
    pass


@dataclass
class SecClient:
    user_agent: str
    max_rps: float = 5.0
    timeout: int = 30

    def __post_init__(self):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/html,*/*",
            }
        )
        self._min_interval = 1.0 / max(self.max_rps, 0.1)
        self._last_ts = 0.0

    def _throttle(self) -> None:
        now = time.time()
        elapsed = now - self._last_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_ts = time.time()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type((requests.RequestException, SecHttpError)),
    )
    def get(self, url: str, *, expect: str = "bytes") -> Any:
        self._throttle()

        try:
            resp = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as e:
            raise SecHttpError(f"Request failed: {e}") from e

        # Retry common throttling statuses
        if resp.status_code in (403, 429, 503):
            raise SecHttpError(f"Throttled/blocked (status={resp.status_code}) for {url}")

        if not resp.ok:
            raise SecHttpError(f"HTTP {resp.status_code} for {url}: {resp.text[:1000]}")

        if expect == "json":
            return resp.json()
        if expect == "text":
            return resp.text
        return resp.content

    def cached_get(self, url: str, cache_path: Path, *, expect: str = "bytes") -> Any:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            if expect == "json":
                import json
                return json.loads(cache_path.read_text(encoding="utf-8"))
            if expect == "text":
                return cache_path.read_text(encoding="utf-8", errors="replace")
            return cache_path.read_bytes()

        data = self.get(url, expect=expect)

        if expect == "json":
            import json
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        elif expect == "text":
            cache_path.write_text(data, encoding="utf-8", errors="replace")
        else:
            cache_path.write_bytes(data)

        return data
