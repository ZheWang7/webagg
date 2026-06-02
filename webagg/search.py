import os, httpx
from typing import Protocol
from . import config  # ensures .env is loaded before we read SERPER_API_KEY


class SearchBackend(Protocol):
    def search(self, query: str, k: int = 10) -> list[dict]: ...


class SerperBackend:
    def __init__(self):
        self.key = os.environ["SERPER_API_KEY"]
        self.url = "https://google.serper.dev/search"

    def search(self, query: str, k: int = 10) -> list[dict]:
        r = httpx.post(self.url, headers={"X-API-KEY": self.key},
                       json={"q": query, "num": k}, timeout=30.0)
        r.raise_for_status()
        out = []
        for item in r.json().get("organic", []):
            out.append({"url": item["link"],
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", "")})
        return out
