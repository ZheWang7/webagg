"""Paginated citation indexes as capture occasions (prereg §6.2 / §10).

webaggr.pdf Def. 5: "a formulation q is a concrete search; issuing it at
step t is a capture occasion returning sources covering some records."
Here the concrete search is "page k of index I for target paper P".
PILOT 1 showed why this plug-in exists: every dense HTML citation page was
censored (JS shells, bot walls, paywalls), while the same data is public
through paginated JSON APIs.

Rules this module encodes:
  * one page fetch = one capture occasion, and the page size we request
    bounds its novelty by construction (Assumption 1b, Y_CAP = page size);
  * pages are emitted CURSOR-STYLE by the pipeline — page k+1 exists only
    after page k is read — so frontier credit changes by at most one page
    per step (prereg §10.3);
  * the ORACLE is still denied: a page URL on config.DENY_DOMAINS comes
    back status="denied" exactly like any other fetch;
  * the logged URL never contains an API key.

Pre-registered enumerators (both free): OpenAlex and Europe PMC.
Semantic Scholar is available as an optional third (needs S2_API_KEY).
The oracle (OpenCitations) is NOT here and must never be added here.
"""
from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import requests

from .canon import canonical_doi
from .fetch import domain_of, is_denied
from .types import Source, sha16

_UA = {"User-Agent": "webagg research agent (Exp 2-C citation enumeration)",
       "Accept": "application/json"}


@dataclass
class Target:
    """The paper whose citers the run collects (the query's subject)."""
    title: str
    doi: str
    pmid: str | None = None       # needed by Europe PMC


@dataclass
class IndexPage:
    """One parsed index page: the rows it lists and where the walk goes next."""
    rows: list = field(default_factory=list)    # [{"title":..., "doi":...}]
    next_cursor: str | None = None              # None = walk finished
    total: int | None = None                    # index's own count (a yield
    #                                             estimate ONLY, never a claim)


class IndexUnavailable(RuntimeError):
    """The index cannot enumerate this target (e.g. no PMID for Europe PMC)."""


class CitationIndex:
    """Base class: paced GET, retries on transient errors, loud otherwise."""

    name = "index"
    first_cursor = "*"

    def __init__(self, client=None, pace_seconds: float = 0.25,
                 retries: int = 3, cooldown: float = 5.0):
        self.client = client or requests          # tests inject a fake
        self.pace, self.retries, self.cooldown = pace_seconds, retries, cooldown
        self._last = 0.0

    # --- subclasses fill these in ---------------------------------------
    def resolve(self, target: Target) -> str:
        """-> the index's own handle for the target paper."""
        raise NotImplementedError

    def request(self, handle: str, cursor: str, size: int, cutoff: str):
        """-> (base_url, public_params, private_params). private_params
        (API keys) are sent but never logged."""
        raise NotImplementedError

    def parse(self, data: dict, cursor: str, cutoff: str) -> IndexPage:
        raise NotImplementedError

    def headers(self) -> dict:
        """Extra request headers (API keys that travel as headers)."""
        return {}

    # --- shared machinery ------------------------------------------------
    def _get(self, url: str, params: dict):
        """-> (http_status or 0, json or None). Transient failures (429,
        5xx, timeouts) are retried with a cooldown; never raises."""
        status = 0
        for attempt in range(self.retries):
            wait = self.pace - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.client.get(url, params=params,
                                       headers={**_UA, **self.headers()},
                                       timeout=(10, 60))
            except requests.RequestException:
                status = 0
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        return 200, resp.json()
                    except ValueError:
                        return 200, None          # 200 but not JSON
                if status not in (429, 500, 502, 503, 504):
                    return status, None           # not transient: give up
            if attempt + 1 < self.retries:
                time.sleep(self.cooldown * (attempt + 1))
        return status, None

    def page_url(self, handle: str, cursor: str, size: int, cutoff: str) -> str:
        """The keyless URL written to the ledger (unique per page)."""
        base, public, _private = self.request(handle, cursor, size, cutoff)
        return f"{base}?{urlencode(public)}"

    def fetch_page(self, target: Target, handle: str, cursor: str, size: int,
                   cutoff: str, formulation_id: str):
        """Fetch + parse one page. -> (Source, IndexPage or None).
        The Source carries the page rendered as text, so the reader and the
        ledger see an index page like any other page. Never raises: the
        outcome is data (a censored page must be visible as censored)."""
        base, public, private = self.request(handle, cursor, size, cutoff)
        url = f"{base}?{urlencode(public)}"
        now = datetime.datetime.utcnow().isoformat()
        src = Source(source_id=sha16(f"{url}|{now}"), url=url,
                     domain=domain_of(url), fetch_time=now,
                     formulation_id=formulation_id, status="error",
                     source_class="citation_index")
        if is_denied(url):                 # the withheld oracle stays withheld
            src.status = "denied"
            return src, None
        status, data = self._get(base, {**public, **private})
        if status in (401, 402, 403):
            src.status = "paywall"         # access censored (bad/missing key)
            return src, None
        if status != 200 or data is None:
            return src, None               # status stays "error"
        try:
            page = self.parse(data, cursor, cutoff)
        except (KeyError, TypeError, AttributeError):
            return src, None               # schema surprise: an error, not data
        page.rows = page.rows[:size]       # the page-size bound is ENFORCED
        if not page.rows:
            src.status = "empty"           # past the end of the walk
            return src, page
        src.status = "ok"
        src.main_text = render_page(self.name, target.title, page)
        return src, page


def render_page(index_name: str, target_title: str, page: IndexPage) -> str:
    """The page as text: a header naming the target (the relevance gate and
    the ER guard read it) and one line per citing paper."""
    lines = [f"Citation index: {index_name}. Papers citing: {target_title}",
             f"Cited paper: {target_title}"]
    for r in page.rows:
        lines.append(f"- {str(r.get('title') or '')[:200]} "
                     f"|| doi: {r.get('doi') or ''}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
class OpenAlexIndex(CitationIndex):
    """OpenAlex works API (cursor paging, free key: OPENALEX_API_KEY)."""

    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def _key(self) -> dict:
        key = os.environ.get("OPENALEX_API_KEY")
        return {"api_key": key} if key else {}

    def resolve(self, target: Target) -> str:
        doi = canonical_doi(target.doi)
        if not doi:
            raise IndexUnavailable(f"OpenAlex needs the target's DOI "
                                   f"(got {target.doi!r})")
        status, data = self._get(self.BASE, {
            "filter": f"doi:https://doi.org/{doi}", "select": "id",
            "per-page": 1, **self._key()})
        results = (data or {}).get("results") or []
        if status != 200 or not results:
            raise IndexUnavailable(f"OpenAlex could not resolve DOI {doi} "
                                   f"(HTTP {status})")
        return str(results[0]["id"]).rsplit("/", 1)[-1]      # 'W123...'

    def request(self, handle, cursor, size, cutoff):
        public = {"filter": (f"cites:{handle},"
                             f"to_publication_date:{cutoff},has_doi:true"),
                  "select": "id,doi,display_name,publication_year",
                  "per-page": size, "cursor": cursor}
        return self.BASE, public, self._key()

    def parse(self, data, cursor, cutoff):
        meta = data.get("meta") or {}
        rows = [{"title": w.get("display_name") or "",
                 "doi": w.get("doi") or ""}
                for w in data.get("results") or []]
        nxt = meta.get("next_cursor") if rows else None
        return IndexPage(rows=rows, next_cursor=nxt, total=meta.get("count"))


class EuropePMCIndex(CitationIndex):
    """Europe PMC search API, CITES: query (cursorMark paging, no key).
    The /citations endpoint is NOT used: its entries carry no DOI
    (measured by scripts/probe_enumerators.py)."""

    name = "europepmc"
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def resolve(self, target: Target) -> str:
        if target.pmid:
            return str(target.pmid).strip()
        doi = canonical_doi(target.doi)
        status, data = self._get(self.BASE, {
            "query": f'DOI:"{doi}"', "resultType": "lite",
            "pageSize": 1, "format": "json"})
        hits = ((data or {}).get("resultList") or {}).get("result") or []
        pmid = hits[0].get("pmid") if hits else None
        if status != 200 or not pmid:
            raise IndexUnavailable(f"Europe PMC has no PMID for DOI {doi} "
                                   f"(HTTP {status}) — not a biomedical "
                                   "paper, or give the PMID in the cohort file")
        return str(pmid)

    def request(self, handle, cursor, size, cutoff):
        public = {"query": (f"CITES:{handle}_MED AND "
                            f"FIRST_PDATE:[1900-01-01 TO {cutoff}]"),
                  "resultType": "lite", "pageSize": size,
                  "cursorMark": cursor, "format": "json"}
        return self.BASE, public, {}

    def parse(self, data, cursor, cutoff):
        hits = (data.get("resultList") or {}).get("result") or []
        rows = [{"title": h.get("title") or "", "doi": h.get("doi") or ""}
                for h in hits]
        nxt = data.get("nextCursorMark")
        if not rows or not nxt or nxt == cursor:
            nxt = None                     # Solr repeats the mark at the end
        return IndexPage(rows=rows, next_cursor=nxt, total=data.get("hitCount"))


class SemanticScholarIndex(CitationIndex):
    """Semantic Scholar graph API (offset paging; S2_API_KEY, 1 request/s).
    OPTIONAL third enumerator: it has no server-side date filter, so rows
    after the cutoff year are dropped here, and its pagination depth is
    unmeasured (scripts/probe_enumerators.py reports it)."""

    name = "semanticscholar"
    first_cursor = "0"
    BASE = "https://api.semanticscholar.org/graph/v1/paper"

    def __init__(self, **kw):
        kw.setdefault("pace_seconds", 1.1)        # the keyed rate limit
        super().__init__(**kw)

    def headers(self) -> dict:
        key = os.environ.get("S2_API_KEY")
        return {"x-api-key": key} if key else {}

    def resolve(self, target: Target) -> str:
        doi = canonical_doi(target.doi)
        if not doi:
            raise IndexUnavailable("Semantic Scholar needs the target's DOI")
        if not os.environ.get("S2_API_KEY"):
            raise IndexUnavailable("S2_API_KEY not set — the anonymous pool "
                                   "answers 429 almost constantly")
        return f"DOI:{doi}"

    def request(self, handle, cursor, size, cutoff):
        public = {"fields": "externalIds,title,year", "limit": size,
                  "offset": cursor}
        return f"{self.BASE}/{handle}/citations", public, {}

    def parse(self, data, cursor, cutoff):
        year_max = int(cutoff[:4])
        entries = data.get("data") or []
        rows = []
        for e in entries:
            cp = e.get("citingPaper") or {}
            if cp.get("year") and int(cp["year"]) > year_max:
                continue                   # published after the cutoff
            rows.append({"title": cp.get("title") or "",
                         "doi": (cp.get("externalIds") or {}).get("DOI") or ""})
        nxt = data.get("next")
        # the cursor follows the API's ENTRIES, not our filtered rows: a
        # page whose rows were all after the cutoff is not the end of the walk
        return IndexPage(rows=rows, total=None,
                         next_cursor=str(nxt) if entries and nxt else None)


INDEXES = {"openalex": OpenAlexIndex, "europepmc": EuropePMCIndex,
           "semanticscholar": SemanticScholarIndex}


def make_indexes(names=("openalex", "europepmc")) -> list:
    """Build enumerators by name; the default pair is the pre-registered
    one (prereg §6.2). The oracle can never be built here."""
    unknown = [n for n in names if n not in INDEXES]
    if unknown:
        raise ValueError(f"unknown index {unknown}; choose from "
                         f"{sorted(INDEXES)}")
    return [INDEXES[n]() for n in names]
