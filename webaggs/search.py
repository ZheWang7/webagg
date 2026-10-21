"""Search backend (Serper). Injectable so tests and replays never hit the web."""
from __future__ import annotations

import os
import time

import requests

from . import config


class SearchError(RuntimeError):
    pass


class SerperBackend:
    """One search issuance = one capture occasion (webaggr.pdf §2.3 Def. 4)."""

    URL = "https://google.serper.dev/search"

    def __init__(self):
        self.key = os.environ.get(config.SERPER_API_KEY_ENV)
        if not self.key:
            raise SearchError(f"{config.SERPER_API_KEY_ENV} not set — refusing "
                              "to run live (loud failure, no dummy results).")

    def search(self, query: str, k: int = 10) -> list[dict]:
        for attempt in (1, 2):
            resp = requests.post(self.URL,
                                 headers={"X-API-KEY": self.key},
                                 json={"q": query, "num": k}, timeout=30)
            if resp.status_code in (429, 500, 502, 503) and attempt == 1:
                time.sleep(10)               # one paced retry on transient
                continue                     # errors, then loud failure
            resp.raise_for_status()
            organic = resp.json().get("organic", [])
            return [{"url": r["link"], "title": r.get("title", "")}
                    for r in organic[:k] if "link" in r]
        raise SearchError("unreachable")
