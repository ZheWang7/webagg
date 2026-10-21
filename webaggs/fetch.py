"""Polite fetcher with the oracle deny list and a fetch-outcome classifier.

Two jobs beyond plain HTTP:
  1. DENY the oracle at run time (withheld-registry design, webaggr.pdf §9):
     any URL on config.DENY_DOMAINS comes back status="denied" with no text,
     and the denial is logged — the agent must reassemble truth from the
     rest of the web.
  2. Classify every fetch outcome (ok / paywall / empty / error). This is the
     access-censoring instrument motivated by the finance Exp 2 diagnosis:
     a paywall stub must be visible as a CENSORED observation in the ledger,
     not disguised as an informative "nothing new here" sample.
"""
from __future__ import annotations

import datetime
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

from . import config
from .types import Source, sha16

_UA = {"User-Agent": "Mozilla/5.0 (research agent; webagg-sports Exp2-S)"}

# Markers that identify a subscription/login stub rather than content.
_PAYWALL_MARKERS = re.compile(
    r"(subscribe to continue|subscription required|sign in to read|"
    r"create a free account|this content is for subscribers|paywall)",
    re.IGNORECASE)

_SKIP_TAGS = {"script", "style", "noscript", "svg", "head"}


class _TextExtractor(HTMLParser):
    """Minimal html -> text (stdlib only; tables flatten to space-joined cells)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0 and data.strip():
            self.chunks.append(data.strip())


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        pass  # malformed HTML: keep whatever parsed; classifier sees the length
    return " ".join(p.chunks)


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def is_denied(url: str) -> bool:
    d = domain_of(url)
    return any(d == dd or d.endswith("." + dd) for dd in config.DENY_DOMAINS)


def classify_source(url: str) -> str:
    """Rule-based reliability class (webaggr.pdf §4; impl guide §10.2 analogue)."""
    d = domain_of(url)
    if any(v in d for v in ("basketball-reference", "espn", "statmuse",
                            "landofbasketball", "realgm", "foxsports")):
        return "stats_vendor"
    if any(n in d for n in ("si.com", "cbssports", "theathletic", "yahoo",
                            "bleacherreport", "apnews", "reuters", "nytimes")):
        return "news"
    return "other"


def fetch_url(url: str, formulation_id: str, *, client=None) -> Source:
    """Fetch one URL and classify the outcome. Never raises on a bad page:
    the outcome IS the data (censoring must be observable, not an exception)."""
    now = datetime.datetime.utcnow().isoformat()
    src = Source(source_id=sha16(f"{url}|{now}"), url=url, domain=domain_of(url),
                 fetch_time=now, formulation_id=formulation_id, status="error",
                 source_class=classify_source(url))

    if is_denied(url):
        # The withheld oracle: refuse loudly, record the denial, return no text.
        src.status = "denied"
        return src

    try:
        get = client.get if client is not None else requests.get
        resp = get(url, headers=_UA, timeout=20)
        if resp.status_code in (401, 402, 403):
            src.status = "paywall"            # access censored by the site
            return src
        resp.raise_for_status()
        text = html_to_text(resp.text)
    except Exception:
        src.status = "error"
        return src

    if _PAYWALL_MARKERS.search(text) and len(text) < 2500:
        src.status = "paywall"                # stub page: censored observation
    elif len(text) < 200:
        src.status = "empty"
    else:
        src.status = "ok"
        src.main_text = text[:40000]          # cap page text fed to the reader
    return src
