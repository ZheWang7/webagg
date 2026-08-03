"""Fetch wrapper (impl guide ch. 5): HTTP GET -> Source | None.

SIGMOD-version requirements implemented here:
  1. non-content (bad status, non-HTML, too-short extraction) returns None
     so it never counts as a fetch in cost denominators or as coverage;
  2. publish_time parsed robustly with dateparser (trafilatura's meta date
     is a STRING in whatever format the page used) and normalized to the
     project's UTC-NAIVE datetime convention;
  3. cache by URL within a run: the same URL surfaced by two formulations
     is fetched once and stays ONE Source (one source_id) -- refetching
     would mint a second source_id (hash of url|fetch_time) and the same
     page would masquerade as two witnesses in corroboration;
  4. the §4.1 Source fields are populated at fetch time: source_class via
     fragmentation.classify(), identity_anchored for regulatory sources.
     (authority_chain_id / doc_type stay None here: registry drivers fill
     them in a later chapter.)
  5. (Sec. 15) the withheld-registry denylist is enforced HERE as the last
     line of defense: denied URLs are refused before the network, and the
     post-redirect final URL is checked too. The search-result layer in
     the pipeline catches denied results earlier; this guard exists so no
     other caller of fetch_url -- present or future -- can bypass the
     denial.
"""
import httpx, trafilatura, dateparser
from datetime import datetime
from urllib.parse import urlparse
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential
from .denylist import get_denylist
from .type_defs import Source
from . import config

UA = config.USER_AGENT

# Run-scoped URL cache (guide ch. 5). Maps url -> Source | None; misses
# (None) are cached too, so a dead URL isn't re-tried by every formulation
# that surfaces it. Cleared by the pipeline at the start of each run.
_CACHE: dict[str, Source | None] = {}


def clear_fetch_cache():
    """Called by the pipeline at run start (cache is per-run, like the DB)."""
    _CACHE.clear()


def _parse_publish_time(meta) -> datetime | None:
    """trafilatura meta.date -> UTC-naive datetime (or None).

    dateparser handles the messy formats real pages use ("Jan 5, 2026",
    "2026-01-05T09:00:00+01:00", ...). tzinfo is stripped to keep the
    project-wide UTC-naive convention (SQLite drops tzinfo anyway, and
    mixing aware/naive datetimes breaks derivation-edge comparisons).
    """
    if not (meta and meta.date):
        return None
    dt = dateparser.parse(str(meta.date))
    return dt.replace(tzinfo=None) if dt else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10),
       reraise=True)
def _get(url: str) -> httpx.Response:
    # retry wraps ONLY the network call: a 404 or paywall is a fact about
    # the page, not a transient error, and must not be retried.
    # reraise=True: when all attempts fail, surface the ORIGINAL httpx
    # exception, not tenacity's RetryError wrapper -- fetch_url's
    # except-net below is typed on httpx errors, and a run-killing
    # RetryError leak is exactly what the first live ch.-14 run hit.
    return httpx.get(url, headers={"User-Agent": UA},
                     follow_redirects=True, timeout=20.0)


def fetch_url(url: str, formulation_id: str) -> Source | None:
    # 0. the withheld-registry denylist (Sec. 15): checked BEFORE the cache
    #    and BEFORE the network. The experiment's claim is "the agent never
    #    read the registry" -- so not one byte, and not even a cached one
    #    (belt-and-braces: the cache is per-run, but the invariant should
    #    not depend on that).
    dl = get_denylist()
    if dl.blocks(url):
        dl.record(url, "fetch")
        _CACHE[url] = None
        return None

    # 1. cache hit (positive or negative): one URL = at most one Source per
    #    run. NOTE: the cached Source keeps the formulation_id of whichever
    #    formulation fetched it FIRST -- first-discoverer attribution.
    if url in _CACHE:
        return _CACHE[url]

    try:
        r = _get(url)
    except (httpx.HTTPError, RetryError):
        # unreachable / timed out after retries: ONE dead URL is a cached
        # miss, never a run-killer (the Exp-1 lesson, now in the pipeline
        # itself, not just the demo shims). RetryError kept in the net as
        # belt-and-braces should the reraise policy ever change.
        _CACHE[url] = None
        return None

    # 0b. post-redirect check (Sec. 15): an allowed link that redirects
    #     INTO a denied domain is still the registry -- the FINAL URL
    #     decides, not the one we were handed. (The body was received but
    #     is discarded unread: it never becomes a Source, a rejection row,
    #     or extractor input.)
    if dl.blocks(str(r.url)):
        dl.record(f"{url} -> {r.url}", "fetch_redirect")
        _CACHE[url] = None
        return None

    # 2. non-content filter (guide ch. 5): never counts as a fetch.
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
        _CACHE[url] = None
        return None
    main_text = trafilatura.extract(r.text) or ""
    if len(main_text) < 200:               # login wall / paywall / stub page
        _CACHE[url] = None
        return None

    meta = trafilatura.extract_metadata(r.text)
    fetch_time = datetime.utcnow()          # UTC-naive convention
    src = Source(
        source_id=Source.make_id(url, fetch_time),
        url=url, domain=urlparse(url).netloc,
        fetch_time=fetch_time,
        publish_time=_parse_publish_time(meta),
        title=meta.title if meta else None,
        main_text=main_text[:20000],        # cap for LLM context
        formulation_id=formulation_id,
    )

    # 3. populate the §4.1 routing/reliability fields at birth.
    from .fragmentation import classify, SourceClass   # lazy: no import cycle
    src.source_class = classify(src).value
    # identity_anchored gates the adversarial cap q-bar (paper §4.4):
    # regulatory registries are anchored. PROVISIONAL: known publishers and
    # the entity's own domain also qualify, but recognizing "own domain"
    # needs the entity context -- the registry-driver chapter refines this.
    src.identity_anchored = (classify(src) is SourceClass.REGULATORY)

    _CACHE[url] = src
    return src
