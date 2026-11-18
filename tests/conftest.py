"""Shared TEST-ONLY fakes, citation domain. Experiment scripts never import
from here — no fakes in any experiment path; tests inject via the
`llm`, `fetch`, `search`, `indexes` parameters.

Two synthetic worlds:
  * TARGET (small): 3 DOI-registered citing papers across 3 domains —
    exercises dedup, the DOI-universe validator, contamination; ends
    ABANDONED (claims are off in this domain and psi is huge at tiny n).
  * BIGPAPER (large): 408 citing papers enumerated by TWO fake paginated
    indexes, 2 per page (y_cap = page size = 2). With tau_max = 420 and
    eps = 0.5 the design arithmetic (webaggs/design.py) gives a feasible
    band of N in (401, 414): both walks finish, every record is doubly
    covered, psi ~= 0.49 < 0.5, and the STATISTICAL certificate fires end
    to end: the path Exp 2-C exists for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs.conformal import ConformalGate            # noqa: E402
from webaggs.indexes import CitationIndex, IndexPage, Target  # noqa: E402
from webaggs.types import Source, sha16                # noqa: E402

TARGET = "Testo Attention Nets: A Sample Paper"
# truth keys exactly as canon.cite_key_from_parts produces them
TRUTH = {
    "doi:10.1234/aaa.1": dict(title="Downstream Applications of Testo Nets",
                              doi="10.1234/aaa.1"),
    "doi:10.1234/bbb.2": dict(title="A Survey of Sample Methods",
                              doi="https://doi.org/10.1234/BBB.2"),
    "doi:10.1234/ccc.3": dict(title="Testo Nets Considered Helpful",
                              doi="10.1234/ccc.3"),
}

PAGES = {
    "https://scholar-mirror.test/testo": {
        "citations": [
            dict(title="Downstream Applications of Testo Nets",
                 doi="10.1234/aaa.1", cited_paper=TARGET, confidence=0.95),
            dict(title="A Survey of Sample Methods",
                 doi="doi.org/10.1234/bbb.2", cited_paper=TARGET,
                 confidence=0.93),
        ]},
    "https://reviews.test/testo-survey": {
        "citations": [
            dict(title="Testo Nets Considered Helpful", doi="10.1234/ccc.3",
                 cited_paper=TARGET, confidence=0.9),
            # NO DOI: outside the pre-registered DOI universe -> rejected
            dict(title="An Unregistered Workshop Note On Testo", doi="",
                 cited_paper=TARGET, confidence=0.9),
            # DUPLICATE of aaa.1 via doi URL form: must dedup to one record
            dict(title="Downstream Applications of Testo Nets (preprint)",
                 doi="https://doi.org/10.1234/AAA.1", cited_paper=TARGET,
                 confidence=0.88),
        ]},
    "https://blog.test/reading-list": {
        "citations": [
            # CONTAMINATION: cites a different paper — ER guard rejects
            dict(title="Unrelated Gradient Tricks", doi="10.9999/zz.9",
                 cited_paper="Some Other Famous Paper", confidence=0.95),
        ]},
}

# --- the large world: statistical certification end to end ---------------
BIGPAPER = "Big Sample Transformers: A Landmark Paper"
BIG_N, BIG_PAGE, BIG_TAU, BIG_EPS = 408, 2, 420, 0.5
BIG_TRUTH = {f"doi:10.5555/big.{i}": dict(title=f"Follow-up Study {i}",
                                          doi=f"10.5555/big.{i}")
             for i in range(BIG_N)}
BIG_CUTOFF = "2023-12-31"
# one prose page that re-mentions two already-indexed citers
BIG_PAGES = {
    "https://blog.test/big-notes": {
        "citations": [dict(title="Follow-up Study 0", doi="10.5555/big.0",
                           cited_paper=BIGPAPER, confidence=0.95),
                      dict(title="Follow-up Study 1", doi="10.5555/big.1",
                           cited_paper=BIGPAPER, confidence=0.95)]},
}


class FakeIndex(CitationIndex):
    """An in-memory paginated citation index. Only the transport (_get)
    is faked: URL building, the deny check, page-size enforcement,
    rendering and the Source all run through the REAL CitationIndex."""

    first_cursor = "0"

    def __init__(self, name, dois, domain=None, fail_once=()):
        super().__init__(pace_seconds=0, retries=1, cooldown=0)
        self.name = name
        self.dois = list(dois)
        self.base = f"https://{domain or name + '.test'}/cites"
        self.fail_once = set(fail_once)        # cursor offsets that 503 once
        self.calls = 0

    def resolve(self, target):
        return "H1"

    def request(self, handle, cursor, size, cutoff):
        return self.base, {"h": handle, "cursor": cursor, "size": size,
                           "to": cutoff}, {}

    def _get(self, url, params):
        self.calls += 1
        off, size = int(params["cursor"]), int(params["size"])
        if off in self.fail_once:
            self.fail_once.discard(off)
            return 503, None
        nxt = off + size if off + size < len(self.dois) else None
        return 200, {"rows": self.dois[off:off + size], "next": nxt,
                     "total": len(self.dois)}

    def parse(self, data, cursor, cutoff):
        rows = [{"title": f"Follow-up Study {d.rsplit('.', 1)[-1]}", "doi": d}
                for d in data["rows"]]
        nxt = None if data["next"] is None else str(data["next"])
        return IndexPage(rows=rows, next_cursor=nxt, total=data["total"])


def big_indexes(overlap: float = 1.0):
    """Two fake indexes over the BIG world. `overlap` < 1 drops the tail of
    index B, leaving those records as singletons (f1 stays up)."""
    dois = [v["doi"] for v in BIG_TRUTH.values()]
    keep = int(round(len(dois) * overlap))
    return [FakeIndex("idx-a", dois), FakeIndex("idx-b", dois[:keep])]


BIG_TARGET = Target(title=BIGPAPER, doi="10.5555/bigpaper", pmid="1")


def big_kwargs(overlap: float = 1.0, **over):
    """The run_query kwargs that put the BIG world on the index path."""
    kw = dict(target=BIG_TARGET, indexes=big_indexes(overlap),
              cutoff=BIG_CUTOFF, page_size=BIG_PAGE, y_cap=BIG_PAGE,
              eps_g=BIG_EPS, max_steps=BIG_TAU)
    kw.update(over)
    return kw


# --- a paper whose normalized title exceeds 160 characters ----------------
# Regression for the Exp 2-C replay bug: record keys in the ledger carry a
# title prefix truncated at 160 characters, the truth table's paper key is
# not truncated, so joining on full keys gave recall 0 for long titles.
LONGPAPER = ("Standards And Guidelines For The Interpretation Of Testo "
             "Variants A Joint Consensus Recommendation Of The Sample College "
             "Of Testing Genetics And Genomics And The Association For "
             "Synthetic Pathology And Fixtures")
LONG_TRUTH = dict(TRUTH)
LONG_PAGES = {url: {"citations": [dict(row, cited_paper=LONGPAPER)
                                  if row.get("cited_paper") == TARGET else row
                                  for row in page["citations"]]}
              for url, page in PAGES.items()}


class RoutedSearch:
    """Query-routed fake search: first matching keyword wins, else default."""

    def __init__(self, routes=(), default=()):
        self.routes = list(routes)
        self.default = list(default) or list(PAGES)
        self.calls = 0

    def search(self, query, k=10):
        self.calls += 1
        q = query.lower()
        for kw, urls in self.routes:
            if kw in q:
                return [{"url": u, "title": ""} for u in urls]
        return [{"url": u, "title": ""} for u in self.default]


class FakeSearch(RoutedSearch):
    def __init__(self):
        super().__init__(routes=(), default=list(PAGES))


def make_fetch(target=TARGET, pages=None):
    """Fake fetch honoring the REAL deny list and per-page paywall status."""
    def _fetch(url, formulation_id):
        import datetime
        from webaggs.fetch import classify_source, domain_of, is_denied
        now = datetime.datetime.utcnow().isoformat()
        src = Source(source_id=sha16(f"{url}|{now}"), url=url,
                     domain=domain_of(url), fetch_time=now,
                     formulation_id=formulation_id, status="ok",
                     source_class=classify_source(url))
        if is_denied(url):
            src.status = "denied"
            return src
        if pages is not None and pages.get(url, {}).get("status") == "paywall":
            src.status = "paywall"
            return src
        # page text names the target so is_relevant passes, plus a marker
        # the fake llm uses to find which page it is reading
        src.main_text = f"Citations of {target}. page-marker {url}"
        return src
    return _fetch


fake_fetch = make_fetch()


def make_fake_llm(pages=PAGES, target=TARGET):
    """Routes by purpose, mimicking the real prompts' JSON contracts."""
    def fake_llm(system, user, *, ledger=None, run_id="", purpose=""):
        if ledger is not None:
            ledger.log_llm(run_id, purpose, "{}", "fake")
        if purpose == "seed":
            return {"formulations": [
                {"query": f"{target.lower()} cited by", "p_success": 0.9,
                 "yield_if_success": 3},
                {"query": f"{target.lower()} survey citations",
                 "p_success": 0.6, "yield_if_success": 1}]}
        if purpose == "followups":
            return {"formulations": []}
        if purpose == "extract":
            url = next((u for u in pages if f"page-marker {u}" in user), None)
            if url is None:
                return {"citations": []}
            return {"citations": list(pages[url]["citations"])}
        raise AssertionError(f"unexpected llm purpose {purpose!r}")
    return fake_llm


def _calibration_pairs():
    return [(0.8 + 0.008 * i, True) for i in range(25)]


def seed_selftest_cohort(data_dir) -> str:
    """OFFLINE self-test cohort under `data_dir`: frozen truth, fitted
    calibration file, and one finished ledger per paper via the real
    end_to_end over the synthetic worlds. BIGPAPER certifies
    STATISTICALLY via the two-index walk (eps=0.5); TARGET ends
    abandoned."""
    import json

    from webaggs.canon import normalize_name
    from webaggs.pipeline import end_to_end
    from webaggs.storage import Ledger
    from webaggs.truth import write_truth

    data_dir = Path(data_dir)
    cohort = "cite_selftest"
    rows = []
    for pid, (paper, truth) in enumerate(((TARGET, TRUTH),
                                          (BIGPAPER, BIG_TRUTH),
                                          (LONGPAPER, LONG_TRUTH)), start=1):
        for key in truth:
            rows.append({"player": paper, "player_id": pid,
                         "season": key, "pts": 1, "gp": 2020})
    write_truth(data_dir, cohort, rows)

    cal = data_dir / "conformal" / "calibration.csv"
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text("confidence,correct\n" +
                   "".join(f"{c},{int(ok)}\n" for c, ok in _calibration_pairs()),
                   encoding="utf-8")

    gate = ConformalGate.fit(_calibration_pairs())
    runs_dir = data_dir / "experiments" / "exp2c" / "runs"
    idx = {}
    for paper, pages, kw in (
            (TARGET, PAGES, dict(eps_g=BIG_EPS, max_steps=30)),
            (LONGPAPER, LONG_PAGES, dict(eps_g=BIG_EPS, max_steps=30)),
            (BIGPAPER, BIG_PAGES, big_kwargs())):
        rid = f"exp2c_{cohort}_{normalize_name(paper).replace(' ', '_')[:30]}"
        ledger = Ledger(runs_dir / f"{rid}.sqlite")
        ans = end_to_end(paper, rid, ledger,
                         search=RoutedSearch(default=list(pages)), gate=gate,
                         llm=make_fake_llm(pages, paper),
                         fetch=make_fetch(paper, pages),
                         delta_m=0.05, budget_usd=50.0, **kw)
        ledger.close()
        idx[normalize_name(paper)] = {
            "done": True, "run_id": rid, "stop_reason": ans.stop_reason,
            "certificate": ans.certificate, "total": ans.total,
            "n_records": ans.n_records}
    (runs_dir.parent / "runs_index.json").write_text(json.dumps(idx, indent=2),
                                                     encoding="utf-8")
    return cohort


@pytest.fixture
def gate():
    return ConformalGate.fit(_calibration_pairs())


@pytest.fixture
def fake_llm():
    return make_fake_llm()


@pytest.fixture
def fake_search():
    return FakeSearch()
