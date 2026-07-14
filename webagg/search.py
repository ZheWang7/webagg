"""Search wrapper (impl guide ch. 5): one interface, one backend.

This is the query interface g(.) of the design paper (§2): ONE search call
= ONE capture occasion for the stopping-rule statistics. The SIGMOD
requirement implemented here: every result is tagged with the formulation
id that surfaced it, so coverage can be attributed to formulations even
for results we never end up fetching (deduplicated or filtered ones still
count as that formulation's discoveries).
"""
import os, httpx
from typing import Protocol
from . import config  # ensures .env is loaded before we read SERPER_API_KEY


class SearchBackend(Protocol):
    def search(self, query: str, k: int = 10,
               formulation_id: str = "") -> list[dict]: ...


class SerperBackend:
    def __init__(self):
        self.key = os.environ["SERPER_API_KEY"]
        self.url = "https://google.serper.dev/search"

    def search(self, query: str, k: int = 10,
               formulation_id: str = "") -> list[dict]:
        """One call = one capture occasion (paper §2).

        Returns [{url, title, snippet, formulation_id}, ...]. The caller
        (frontier loop) passes the formulation whose query this is; the id
        rides on every result so downstream attribution never has to guess.
        """
        r = httpx.post(self.url, headers={"X-API-KEY": self.key},
                       json={"q": query, "num": k}, timeout=30.0)
        r.raise_for_status()
        out = []
        for item in r.json().get("organic", []):
            out.append({"url": item["link"],
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "formulation_id": formulation_id})  # guide ch. 5
        return out
