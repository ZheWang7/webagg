"""The withheld-registry denylist (impl guide Sec. 15).

The oracle experiment's logic is: build the answer key FROM the registry
(scripts/build_truth.py), then grade what the agent reassembles WITHOUT it
(paper Sec. 9: "deny the agent the registry"). This module is the
"without" -- one suffix predicate enforced at both open-web ingress points:

    search-result layer (pipeline.run_query)  -- denied results are dropped
        before any fetch, so registry bytes never even leave the network;
    fetch layer (fetch.fetch_url)             -- a second gate before the
        GET, plus a post-redirect check, so an allowed link that 301s into
        the registry is still caught.

The oracle itself is untouched BY CONSTRUCTION: the schema drivers run
their own httpx client and never pass through fetch_url, so build_truth's
"the denylist never applies here" is structural, not a flag.

Experiment integrity over convenience, twice:

  * a bare token the alias table does not know RAISES -- a typo like
    `--deny sce` silently denying nothing would poison every conclusion
    drawn from the run;
  * assert_no_denied_sources() re-scans the finished run DB and refuses a
    contaminated run loudly. The denial is enforced online AND verified
    offline on the artifact itself; the grading harness re-runs the same
    check before trusting any run.

Scope note for experiment write-ups: this denies the REGISTRY, exactly as
the guide specifies -- mirrors that republish registry content (they exist
for EDGAR) are open web and stay reachable unless the experimenter adds
their suffixes explicitly. Whatever list is in force is stamped into the
run DB (`denylist_active`) and the result dict, so every run states its
own denial scope.

Per-run module state (set_denylist in run_query) follows the repo's
existing convention for fetch's URL cache and llm's cost logger: one
process-wide slot, re-initialized at every run start so runs cannot leak
into each other.
"""
from __future__ import annotations

from urllib.parse import urlparse

# Bare tokens the CLI accepts (the guide spells the experiment `--deny sec`).
# Anything containing a dot is taken verbatim as a raw domain suffix.
ALIASES: dict[str, tuple[str, ...]] = {
    "sec": ("sec.gov",),                        # EDGAR: www/ data/ efts/ ...
    "clinicaltrials": ("clinicaltrials.gov",),  # the other registry driver
}


def normalize(entries) -> tuple[str, ...]:
    """CLI entries -> lowercase suffix tuple. LOUD on unknown bare tokens."""
    out: list[str] = []
    for e in entries or ():
        e = str(e).strip().lower().lstrip(".")
        if not e:
            continue
        if "." in e:
            out.append(e)                       # raw domain suffix, as given
        elif e in ALIASES:
            out.extend(ALIASES[e])              # the guide's shorthand
        else:
            raise ValueError(
                f"unknown denylist token {e!r}: not a domain (no dot) and "
                f"not an alias in {sorted(ALIASES)}. Refusing to run with "
                f"a denylist that might deny nothing.")
    return tuple(dict.fromkeys(out))            # dedupe, order preserved


class Denylist:
    """A frozen set of domain suffixes + a hit log for run provenance."""

    def __init__(self, entries=()):
        self.suffixes = normalize(entries)
        self.hits: list[dict] = []              # every denial: {url, layer}

    def __bool__(self) -> bool:
        return bool(self.suffixes)

    def blocks(self, url) -> bool:
        """True iff url's host is a denied domain or a subdomain of one.

        Label-wise suffix match on the HOSTNAME: 'sec.gov' blocks sec.gov,
        www.sec.gov and data.sec.gov -- but NOT mysec.gov (different
        label) and NOT sec.gov.evil.com (its suffix is evil.com).
        hostname (not netloc) drops ports and credentials before matching.
        """
        host = (urlparse(str(url)).hostname or "").lower()
        if not host:
            return False
        return any(host == s or host.endswith("." + s)
                   for s in self.suffixes)

    def record(self, url, layer: str) -> None:
        """Log one denial; the pipeline turns these into measurement rows."""
        self.hits.append({"url": str(url), "layer": layer})

    def describe(self) -> dict:
        """The run-provenance stamp: what was denied, how often it was hit."""
        return {"suffixes": list(self.suffixes), "n_denied": len(self.hits)}


# --- the per-run module slot (same pattern as fetch._CACHE) -----------------
_ACTIVE = Denylist(())


def set_denylist(entries=()) -> Denylist:
    """Install this run's denylist (empty = allow everything). Returns it."""
    global _ACTIVE
    _ACTIVE = Denylist(entries)
    return _ACTIVE


def get_denylist() -> Denylist:
    return _ACTIVE


def clear_denylist() -> None:
    set_denylist(())


def assert_no_denied_sources(session, denylist: Denylist | None = None) -> None:
    """Refuse a contaminated run: no row in sources OR rejected_sources may
    sit on a denied domain.

    Rejections count too: a rejected page was still FETCHED and read by
    the relevance filter, and "the agent never saw the registry" must mean
    never -- not "saw it and filtered it". A run that fails here cannot be
    graded against the oracle; better a dead run than a poisoned plot.
    """
    dl = denylist if denylist is not None else get_denylist()
    if not dl:
        return
    from .storage import SourceRow, RejectedSourceRow   # lazy: no cycle
    bad = [r.url for r in session.query(SourceRow).all() if dl.blocks(r.url)]
    bad += [r.url for r in session.query(RejectedSourceRow).all()
            if dl.blocks(r.url)]
    if bad:
        raise RuntimeError(
            "withheld-registry violation: the run DB contains fetched pages "
            f"on denied domains {list(dl.suffixes)}: {bad[:5]}"
            + (" ..." if len(bad) > 5 else ""))
