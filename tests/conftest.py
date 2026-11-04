"""Shared TEST-ONLY fakes, citation domain. Experiment scripts never import
from here — no fakes in any experiment path; tests inject via the
`llm`, `fetch`, `search` parameters.

Two synthetic worlds:
  * TARGET (small): 3 citing papers across 3 domains — exercises dedup,
    DOI-vs-title keying, contamination; ends ABANDONED (claims are off in
    this domain and psi is huge at tiny n).
  * BIGPAPER (large): 400 citing papers fully covered by two index pages —
    with max_steps=4 and eps=0.30, U_hat hits 0 (every record doubly
    covered), psi = (24/400)*sqrt(2*4*ln(1/0.05)) ~= 0.147 < 0.30, and the
    STATISTICAL certificate fires end to end: the path Exp 2-C exists for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs.conformal import ConformalGate            # noqa: E402
from webaggs.types import Source, sha16                # noqa: E402

TARGET = "Testo Attention Nets: A Sample Paper"
# truth keys exactly as canon.cite_key_from_parts produces them
TRUTH = {
    "doi:10.1234/aaa.1": dict(title="Downstream Applications of Testo Nets",
                              doi="10.1234/aaa.1"),
    "doi:10.1234/bbb.2": dict(title="A Survey of Sample Methods",
                              doi="https://doi.org/10.1234/BBB.2"),
    "t:testo nets considered helpful": dict(
        title="Testo Nets Considered Helpful", doi=""),
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
            dict(title="Testo Nets Considered Helpful", doi="",
                 cited_paper=TARGET, confidence=0.9),
            # DUPLICATE of aaa.1 via doi URL form: must dedup to one record
            dict(title="Downstream Applications of Testo Nets (preprint)",
                 doi="https://doi.org/10.1234/AAA.1", cited_paper=TARGET,
                 confidence=0.88),
        ]},
    "https://blog.test/reading-list": {
        "citations": [
            # CONTAMINATION: cites a different paper — ER guard rejects
            dict(title="Unrelated Gradient Tricks", doi="10.9/zz.9",
                 cited_paper="Some Other Famous Paper", confidence=0.95),
        ]},
}

# --- the large world: statistical certification end to end ---------------
BIGPAPER = "Big Sample Transformers: A Landmark Paper"
BIG_TRUTH = {f"doi:10.5555/big.{i}": dict(title=f"Follow-up Study {i}",
                                          doi=f"10.5555/big.{i}")
             for i in range(400)}
BIG_PAGES = {
    "https://scholar-mirror.test/big-1": {
        "citations": [dict(title=v["title"], doi=v["doi"],
                           cited_paper=BIGPAPER, confidence=0.95)
                      for v in BIG_TRUTH.values()]},
    "https://index.test/big-2": {
        "citations": [dict(title=v["title"], doi=v["doi"],
                           cited_paper=BIGPAPER, confidence=0.94)
                      for v in BIG_TRUTH.values()]},
}


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
    STATISTICALLY (max_steps=4, eps=0.30); TARGET ends abandoned."""
    import json

    from webaggs.canon import normalize_name
    from webaggs.pipeline import end_to_end
    from webaggs.storage import Ledger
    from webaggs.truth import write_truth

    data_dir = Path(data_dir)
    cohort = "cite_selftest"
    rows = []
    for pid, (paper, truth) in enumerate(((TARGET, TRUTH),
                                          (BIGPAPER, BIG_TRUTH)), start=1):
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
    for paper, pages, steps in ((TARGET, PAGES, 30), (BIGPAPER, BIG_PAGES, 4)):
        rid = f"exp2c_{cohort}_{normalize_name(paper).replace(' ', '_')[:30]}"
        ledger = Ledger(runs_dir / f"{rid}.sqlite")
        ans = end_to_end(paper, rid, ledger,
                         search=RoutedSearch(default=list(pages)), gate=gate,
                         llm=make_fake_llm(pages, paper),
                         fetch=make_fetch(paper, pages),
                         eps_g=0.30, delta_m=0.05, budget_usd=5.0,
                         max_steps=steps)
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
