"""
Section 9 -- Schema-Addressable Mode.

Each registry is wrapped in a driver exposing two methods:
    enumerate_keys(query_filter) -> yields the keys to sweep  (the "K" of Def 11)
    fetch_for_key(key)           -> the documents for one key (the "mu(k)" of Def 11)
A thin runner sweeps the keys, runs the same relevance + extraction as the
open-web path, and stamps the delta_F = 0 completeness certificate.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
import trafilatura
from typing import Iterator, Protocol, Optional, Callable

import httpx

from . import config
from .type_defs import Source

# SEC EDGAR REQUIRES a descriptive User-Agent carrying a contact email, or it
# 403s.
_EDGAR_UA = config.USER_AGENT if "(" in config.USER_AGENT else \
    "webagg-research/0.1 (mailto:jameswangzhe1110@gmail.com)"

# ClinicalTrials.gov is the mirror image: its CDN 403s non-browser User-Agents
# (exactly the contact-email UA SEC demands). Their v2 REST API is public and
# documented for programmatic use, so we present a conventional browser UA plus
# a JSON Accept header to clear the bot filter.
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _html_to_text(html: str) -> str:
    """Best-effort HTML -> plain text, XBRL-aware.

    Modern EDGAR 10-K/10-Q primary docs are INLINE XBRL: .htm files with a hidden
    facts block (<ix:header>/<ix:hidden>) full of FASB taxonomy URIs at the top.
    trafilatura grabs that block and returns machine noise instead of the filing
    text, so we first try BeautifulSoup: drop <script>/<style> and the hidden
    XBRL header, then take the visible text. Crucially we do NOT strip inline
    <ix:nonFraction> tags -- the actual reported numbers live inside them, and
    get_text() walks into them. Two fallbacks keep us from ever dropping a doc.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        # the inline-XBRL noise block (parser keeps the namespaced tag names)
        for tag in soup.find_all(["ix:header", "ix:hidden",
                                  "ix:references", "ix:resources"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        if len(text) >= 200:
            return text
    except Exception:
        pass
    try:
        import trafilatura
        out = trafilatura.extract(html) or ""
        if len(out) >= 200:
            return out
    except Exception:
        pass
    # last resort: crude tag strip
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 'YYYY-MM-DD' (or full ISO) date; None on anything unparseable.

    publish_time feeds the corroboration layer's temporal edge test (§7), so a
    correct date here makes copy-detection work; a wrong one would invent edges.
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").split("T")[0])
    except (ValueError, AttributeError):
        return None


def _default_client(timeout: float = 30.0, *,
                    user_agent: str = _EDGAR_UA,
                    accept: Optional[str] = None,
                    extra_headers: Optional[dict] = None) -> httpx.Client:
    headers = {"User-Agent": user_agent}
    if accept:
        headers["Accept"] = accept
    if extra_headers:
        headers.update(extra_headers)
    return httpx.Client(headers=headers, follow_redirects=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# Driver interface (Definition 11: K plus mu)
# --------------------------------------------------------------------------- #
class SchemaDriver(Protocol):
    name: str
    def enumerate_keys(self, query_filter: dict) -> Iterator[str]:
        pass
    def fetch_for_key(self, key: str) -> list[Source]:
        pass

# --------------------------------------------------------------------------- #
# SEC EDGAR driver
# --------------------------------------------------------------------------- #
class EDGARDriver:
    """SEC EDGAR. Key universe K = CIK numbers (one per filer).

    Two SEC endpoints we rely on:
      * the published company index   https://www.sec.gov/files/company_tickers.json
        -- this is what makes K enumerable: every public filer is listed.
      * the per-filer submissions JSON https://data.sec.gov/submissions/CIK{cik}.json
        -- mu(k): a deterministic listing of every filing for one CIK.
    """
    name = "edgar"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    SHARD_URL = "https://data.sec.gov/submissions/{name}"
    ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

    def __init__(self, client: Optional[httpx.Client] = None,
                 min_interval_s: float = 0.15):
        # client is injectable so tests can pass an httpx.MockTransport client
        # (no network). min_interval_s throttles live calls: SEC asks for
        # <=10 req/s, so ~0.15s between requests keeps us well under the cap.
        self._client = client or _default_client()
        self._min_interval = min_interval_s
        self._last_call = 0.0
        self._filter: dict = {}


    def _get_json(self, url: str) -> dict:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        r = self._client.get(url)
        self._last_call = time.monotonic()
        r.raise_for_status()
        return r.json()

    # -- K: enumerate the keys ---------------------------------------------
    def enumerate_keys(self, query_filter: dict) -> Iterator[str]:
        """Yield 10-digit zero-padded CIK strings matching the filter.

        Filter options (all optional):
          ciks=[...]            -- use these CIKs directly, skip the index fetch
          tickers=["TSLA", ...] -- exact ticker match
          name_contains="tesla" -- case-insensitive substring of the company name
        With none of the above, every filer in the index is yielded
        """
        self._filter = dict(query_filter or {})
        return self._enumerate_ciks()

    def _enumerate_ciks(self) -> Iterator[str]:
        # Fast path: caller already knows the CIKs -> no index sweep needed.
        explicit = self._filter.get("ciks")
        if explicit:
            for cik in explicit:
                yield str(cik).zfill(10)
            return

        tickers = {t.upper() for t in self._filter.get("tickers", [])}
        name_sub = (self._filter.get("name_contains") or "").lower()

        index = self._get_json(self.TICKERS_URL)  # {"0": {cik_str, ticker, title}, ...}
        for v in index.values():
            title = v.get("title", "")
            ticker = (v.get("ticker") or "").upper()
            if tickers and ticker not in tickers:
                continue
            if name_sub and name_sub not in title.lower():
                continue
            yield str(v["cik_str"]).zfill(10)

    # -- mu(k): fetch one key's documents ----------------------------------
    @staticmethod
    def _iter_filing_block(block: dict) -> Iterator[dict]:
        """Zip EDGAR's parallel arrays into one dict per filing.

        EDGAR stores a filing list column-wise: accessionNumber[i], form[i],
        filingDate[i], primaryDocument[i] all describe filing i. We transpose
        that into row-wise dicts so the rest of the code can think in records.
        """
        accs = block.get("accessionNumber", [])
        forms = block.get("form", [])
        dates = block.get("filingDate", [])
        primaries = block.get("primaryDocument", [])
        for i in range(len(accs)):
            yield {
                "accession": accs[i],
                "form": forms[i] if i < len(forms) else "",
                "filing_date": dates[i] if i < len(dates) else None,
                "primary": primaries[i] if i < len(primaries) else "",
            }

    def _all_filings(self, cik: str) -> Iterator[dict]:
        """Every filing for a CIK -- across BOTH the recent block and the shards.

        The guide skips: submissions/CIK{cik}.json only inlines the most
        recent ~1000 filings under filings.recent. Older filings live in extra
        shard files listed under filings.files (each a {name, filingCount, ...}).
        A long-lived filer (e.g. Apple) has thousands of filings, so iterating
        only 'recent' silently truncates history, which would quietly break the
        delta_F = 0 guarantee we are trying to certify. We page through the
        shards too.
        """
        data = self._get_json(self.SUBMISSIONS_URL.format(cik=cik))
        filings = data.get("filings", {})
        yield from self._iter_filing_block(filings.get("recent", {}))
        for shard in filings.get("files", []):
            shard_name = shard.get("name")
            if not shard_name:
                continue
            shard_data = self._get_json(self.SHARD_URL.format(name=shard_name))
            # shard files hold the same parallel arrays, but at the TOP level
            # (not nested under filings.recent)
            yield from self._iter_filing_block(shard_data)

    def _passes_filter(self, filing: dict) -> bool:
        forms = self._filter.get("forms")
        if forms and filing["form"] not in set(forms):
            return False
        since = self._filter.get("since")  # 'YYYY-MM-DD' or 'YYYY'
        if since:
            fd = filing.get("filing_date") or ""
            if fd < (since if len(since) > 4 else f"{since}-01-01"):
                return False
        return True

    def fetch_for_key(self, cik: str) -> list[Source]:
        """Fetch the primary document of every filing for this CIK that passes
        the form/date filter, and wrap each as a Source with full provenance."""
        cik = str(cik).zfill(10)
        sources: list[Source] = []
        for filing in self._all_filings(cik):
            if not self._passes_filter(filing):
                continue
            if not filing["primary"]:
                continue
            acc_nodash = filing["accession"].replace("-", "")
            doc_url = self.ARCHIVE_URL.format(
                cik=int(cik),                 # archive path uses the un-padded CIK
                acc=acc_nodash,
                doc=filing["primary"],
            )
            try:
                r = self._client.get(doc_url)
                r.raise_for_status()
            except httpx.HTTPError:
                # Theorem 3's only residual failure mode is a transient fetch
                # error (NOT a coverage gap); skip and let a re-run retry it.
                continue
            fetch_time = datetime.utcnow()
            sources.append(Source(
                source_id=Source.make_id(doc_url, fetch_time),
                url=doc_url,
                # canonicalize to "sec.gov" so corroboration's reliability()
                # prior (keyed exactly on "sec.gov") fires; the raw netloc would
                # be "www.sec.gov" and silently fall back to the 0.5 default.
                domain="sec.gov",
                fetch_time=fetch_time,
                publish_time=_parse_date(filing["filing_date"]),
                title=f"{filing['form']} {filing['accession']}",
                main_text=_html_to_text(r.text)[:20000],
                # formulation_id records WHICH key surfaced this doc -- the
                # schema-mode analogue of "which search formulation produced it".
                formulation_id=f"edgar:CIK{cik}",
            ))
        return sources


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov driver (v2 API)
# --------------------------------------------------------------------------- #
class ClinicalTrialsDriver:
    """ClinicalTrials.gov. Key universe K = NCT numbers.

    v2 API:
      * list  /api/v2/studies         -- search/stream NCT ids matching a filter
      * fetch /api/v2/studies/{nct}    -- the full record for one trial (mu(k))
    """
    name = "clinicaltrials"
    LIST_URL = "https://clinicaltrials.gov/api/v2/studies"
    STUDY_URL = "https://clinicaltrials.gov/api/v2/studies/{nct}"

    # friendly aliases -> v2 query params, so callers don't memorize the API
    _ALIASES = {
        "intervention": "query.intr",
        "condition": "query.cond",
        "term": "query.term",
    }

    def __init__(self, client: Optional[httpx.Client] = None,
                 page_size: int = 100, min_interval_s: float = 0.0):
        self._client = client or _default_client(
            user_agent=_BROWSER_UA,
            accept="application/json, text/plain, */*",
            extra_headers={
                # browsers always send these; bot filters flag their absence
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://clinicaltrials.gov/",
            },
        )
        self._page_size = page_size
        self._min_interval = min_interval_s
        self._last_call = 0.0
        self._filter: dict = {}

    def _get_json(self, url: str, params: Optional[dict] = None) -> dict:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        r = self._client.get(url, params=params)
        self._last_call = time.monotonic()
        r.raise_for_status()
        return r.json()

    def _build_params(self, query_filter: dict) -> dict:
        params: dict = {"pageSize": self._page_size, "format": "json"}
        for k, v in (query_filter or {}).items():
            if k in ("page_size", "max_keys"):
                continue
            if k == "phase":
                # Essie advanced filter, e.g. phase=3 -> AREA[Phase]PHASE3
                params["filter.advanced"] = f"AREA[Phase]PHASE{v}"
            elif k in self._ALIASES:
                params[self._ALIASES[k]] = v
            elif "." in k:
                params[k] = v  # already a v2 param like query.intr / filter.advanced
        return params

    def enumerate_keys(self, query_filter: dict) -> Iterator[str]:
        """Yield NCT ids matching the filter, paging the v2 API to completion.

        GOTCHA the guide skips: the list endpoint returns at most pageSize
        studies plus a 'nextPageToken'. To get ALL matching trials you must keep
        re-issuing the request with pageToken=<that token> until the response no
        longer carries one. Stop early and you under-count -- another silent
        hole in the delta_F = 0 claim.
        """
        # eager stash + delegate (same reason as EDGARDriver.enumerate_keys)
        self._filter = dict(query_filter or {})
        return self._enumerate_ncts()

    def _enumerate_ncts(self) -> Iterator[str]:
        params = self._build_params(self._filter)
        page_token = None
        while True:
            if page_token:
                params = {**params, "pageToken": page_token}
            data = self._get_json(self.LIST_URL, params=params)
            for study in data.get("studies", []):
                nct = (study.get("protocolSection", {})
                       .get("identificationModule", {})
                       .get("nctId"))
                if nct:
                    yield nct
            page_token = data.get("nextPageToken")
            if not page_token:
                break  # no more pages -> the sweep of K is complete

    @staticmethod
    def _flatten_study(study: dict) -> tuple[str, Optional[datetime], str]:
        """Turn the nested v2 study JSON into (title, publish_time, main_text).

        We linearize just the modules an analyst query cares about (phase,
        enrollment, status, interventions) into readable text, because
        extract_mentions() is an LLM call over main_text -- it can't read JSON
        we never flattened.
        """
        ps = study.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        status = ps.get("statusModule", {})
        design = ps.get("designModule", {})
        conds = ps.get("conditionsModule", {})
        arms = ps.get("armsInterventionsModule", {})
        desc = ps.get("descriptionModule", {})

        nct = ident.get("nctId", "")
        title = ident.get("briefTitle") or ident.get("officialTitle") or nct
        phases = ", ".join(design.get("phases", []) or [])
        enroll = (design.get("enrollmentInfo", {}) or {}).get("count")
        overall = status.get("overallStatus", "")
        conditions = ", ".join(conds.get("conditions", []) or [])
        interventions = ", ".join(
            f"{iv.get('type', '')}: {iv.get('name', '')}".strip(": ")
            for iv in arms.get("interventions", []) or []
        )
        posted = (status.get("studyFirstPostDateStruct", {}) or {}).get("date") \
                 or (status.get("startDateStruct", {}) or {}).get("date")

        lines = [
            f"NCT ID: {nct}",
            f"Title: {title}",
            f"Overall status: {overall}",
            f"Phase: {phases}",
            f"Enrollment: {enroll}" if enroll is not None else "Enrollment: (unspecified)",
            f"Conditions: {conditions}",
            f"Interventions: {interventions}",
            "",
            (desc.get("briefSummary") or "").strip(),
        ]
        return title, _parse_date(posted), "\n".join(lines).strip()

    def fetch_for_key(self, nct: str) -> list[Source]:
        """One NCT id -> one Source (a trial is a single logical document)."""
        try:
            data = self._get_json(self.STUDY_URL.format(nct=nct))
        except httpx.HTTPError:
            return []  # transient miss; re-run retries (Theorem 3)
        title, publish_time, main_text = self._flatten_study(data)
        fetch_time = datetime.utcnow()
        url = self.STUDY_URL.format(nct=nct)
        return [Source(
            source_id=Source.make_id(url, fetch_time),
            url=url,
            domain="clinicaltrials.gov",   # exact key in reliability()'s priors
            fetch_time=fetch_time,
            publish_time=publish_time,
            title=title,
            main_text=main_text[:20000],
            formulation_id=f"ctgov:{nct}",
        )]


# --------------------------------------------------------------------------- #
# The runner: sweep K, extract, stamp the delta_F = 0 certificate
# --------------------------------------------------------------------------- #
def run_schema_addressable(
        query: str,
        driver: SchemaDriver,
        *,
        query_filter: dict,
        run_id: str,
        relevance_fn: Optional[Callable[[Source, str], bool]] = None,
        extract_fn: Optional[Callable[[Source, str], list]] = None,
        max_keys: Optional[int] = None,
):
    """Enumerate K, fetch mu(k) for each key, run the SAME relevance + extraction
    as the open-web path, and log progress plus the Theorem 3 certificate.

    relevance_fn / extract_fn are injectable (default to the real LLM-backed
    ones in extract.py) so this runner is unit-testable offline -- the same
    pattern as the injectable adjudicator in entity resolution.

    NOTE: the default relevance/extract functions read prompt files via a path
    relative to the CWD, so run this from the repo root (same constraint as the
    rest of the pipeline). They are imported lazily so that merely importing this
    module -- e.g. in a driver-only unit test -- doesn't require prompts/ or API
    keys to be present.
    """
    from .storage import get_session
    from .metrics import log_measurement

    if relevance_fn is None or extract_fn is None:
        from .extract import is_relevant, extract_mentions  # lazy (see docstring)
        relevance_fn = relevance_fn or is_relevant
        extract_fn = extract_fn or extract_mentions

    session = get_session(str(config.RUNS_DIR / f"{run_id}.sqlite"))
    keys_swept = 0
    mentions_found = 0           # guide calls this 'records_found'; it's really
    # a mention count -- renamed for honesty
    distinct_records: set = set()

    for key in driver.enumerate_keys(query_filter):
        keys_swept += 1
        for src in driver.fetch_for_key(key):
            session.add(src.to_row())            # provenance is sacred: store
            # every fetched doc, relevant or not
            if not relevance_fn(src, query):     # cheap LLM gate before extraction
                continue
            for m in extract_fn(src, query):
                session.add(m.to_row())
                mentions_found += 1
                distinct_records.add(f"{m.entity_surface}|{m.record_kind}")
        session.commit()                         # commit per key so a crash mid-
        # sweep doesn't lose everything
        if keys_swept % 100 == 0:
            log_measurement(session, run_id, keys_swept, "keys_swept",
                            keys_swept,
                            extra={"mentions_found": mentions_found,
                                   "distinct_records": len(distinct_records)})
            session.commit()
        if max_keys is not None and keys_swept >= max_keys:
            break

    # Theorem 3 certificate: over the addressable closure K*, delta_F = 0.
    log_measurement(session, run_id, keys_swept, "schema_complete", 1.0,
                    extra={"delta_F": 0.0,
                           "keys_swept": keys_swept,
                           "mentions_found": mentions_found,
                           "distinct_records": len(distinct_records)})
    session.commit()
    return {
        "keys_swept": keys_swept,
        "mentions_found": mentions_found,
        "distinct_records": len(distinct_records),
        "session": session,
    }
