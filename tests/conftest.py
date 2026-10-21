"""Shared TEST-ONLY fakes. Experiment scripts never import from here —
no fakes in any experiment path (methodology rule); tests monkeypatch via
the injectable `llm`, `fetch`, `search` parameters instead.

The synthetic world: player 'Testo Sampleson', 3 NBA seasons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs.conformal import ConformalGate            # noqa: E402
from webaggs.types import Source, sha16                # noqa: E402

PLAYER = "Testo Sampleson"
TRUTH = {"2019-20": 450, "2020-21": 612, "2021-22": 388}   # total 1450

# Three pages on three domains; page C carries the aggregate claims.
PAGES = {
    "https://statsvendor.test/testo": {
        "rows": [("Testo Sampleson", "2019-20", 450, 0.95),
                 ("Testo Sampleson", "2020-21", 612, 0.95)],
        "claims": []},
    "https://news.test/testo-recap": {
        "rows": [("T. Sampleson", "2021-22", 388, 0.9)],
        "claims": [{"kind": "COUNT", "value": 3}]},
    "https://blog.test/testo-career": {
        "rows": [("Testo Sampleson", "2019-20", 450, 0.85),
                 ("Testo Sampleson", "2021-22", 388, 0.85)],
        "claims": [{"kind": "COUNT", "value": 3},
                   {"kind": "SUM", "value": 1450}]},
}

SEARCH_RESULTS = {  # every query surfaces every page (dense little web)
    "default": [{"url": u, "title": "t"} for u in PAGES],
}


class RoutedSearch:
    """Query-routed fake search: first matching keyword wins, else default."""

    def __init__(self, routes=(), default=()):
        self.routes = list(routes)            # [(keyword, [urls])]
        self.default = list(default) or list(PAGES)
        self.calls = 0

    def search(self, query, k=10):
        self.calls += 1
        q = query.lower()
        for kw, urls in self.routes:
            if kw in q:
                return [{"url": u, "title": ""} for u in urls]
        return [{"url": u, "title": ""} for u in self.default]


class FakeSearch(RoutedSearch):               # back-compat: every query -> PAGES
    def __init__(self):
        super().__init__(routes=(), default=list(PAGES))


def make_fetch(player=PLAYER, pages=None):
    """Fake fetch honoring the REAL deny list and per-page paywall statuses."""
    def _fetch(url, formulation_id):
        import datetime
        from webaggs.fetch import classify_source, domain_of, is_denied
        now = datetime.datetime.utcnow().isoformat()
        src = Source(source_id=sha16(f"{url}|{now}"), url=url,
                     domain=domain_of(url), fetch_time=now,
                     formulation_id=formulation_id, status="ok",
                     source_class=classify_source(url))
        if is_denied(url):                     # mirror the oracle deny list
            src.status = "denied"
            return src
        if pages is not None and pages.get(url, {}).get("status") == "paywall":
            src.status = "paywall"
            return src
        src.main_text = f"{player} career page {url}"
        return src
    return _fetch


fake_fetch = make_fetch()                      # back-compat alias


def make_fake_llm(pages=PAGES, player=PLAYER):
    """Routes by purpose, mimicking the real prompts' JSON contracts."""
    def fake_llm(system, user, *, ledger=None, run_id="", purpose=""):
        if ledger is not None:
            ledger.log_llm(run_id, purpose, "{}", "fake")
        if purpose == "seed":
            return {"formulations": [
                {"query": f"{player.lower()} career stats", "p_success": 0.9,
                 "yield_if_success": 3},
                {"query": f"{player.lower()} recent season points",
                 "p_success": 0.6, "yield_if_success": 1}]}
        if purpose == "followups":
            return {"formulations": []}
        if purpose == "extract":
            url = next((u for u in pages if f"career page {u}" in user), None)
            if url is None:
                return {"season_rows": [], "claims": []}
            page = pages[url]
            return {"season_rows": [
                        {"player": p, "season": s, "points": v, "confidence": c}
                        for p, s, v, c in page["rows"]],
                    "claims": page["claims"]}
        raise AssertionError(f"unexpected llm purpose {purpose!r}")
    return fake_llm


# --- a second synthetic player: no claims, one season never on any page ----
BENCHY = "Benchy Marks"
BENCHY_TRUTH = {"2015-16": 200, "2016-17": 350, "2018-19": 120}   # 3 seasons
BENCHY_PAGES = {                               # covers only 2 of the 3
    "https://a.test/benchy": {
        "rows": [("Benchy Marks", "2015-16", 200, 0.9),
                 ("Benchy Marks", "2016-17", 350, 0.9)],
        "claims": []},
}


def _calibration_pairs():
    return [(0.8 + 0.008 * i, True) for i in range(25)]


def seed_selftest_cohort(data_dir) -> str:
    """Build a complete OFFLINE self-test cohort under `data_dir`: frozen
    truth, a fitted conformal calibration file, and one finished ledger per
    player produced by the real end_to_end over the synthetic worlds
    (fakes — tests only, never an experiment path). Returns the cohort name."""
    from pathlib import Path

    from webaggs.canon import normalize_name
    from webaggs.conformal import ConformalGate
    from webaggs.pipeline import end_to_end
    from webaggs.storage import Ledger
    from webaggs.truth import write_truth

    data_dir = Path(data_dir)
    cohort = "nba_selftest"
    rows = []
    for pid, (player, truth) in enumerate(
            ((PLAYER, TRUTH), (BENCHY, BENCHY_TRUTH)), start=1):
        for season, pts in truth.items():
            rows.append({"player": player, "player_id": pid,
                         "season": season, "pts": pts, "gp": 50})
    write_truth(data_dir, cohort, rows)

    cal = data_dir / "conformal" / "calibration.csv"
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text("confidence,correct\n" +
                   "".join(f"{c},{int(ok)}\n" for c, ok in _calibration_pairs()),
                   encoding="utf-8")

    gate = ConformalGate.fit(_calibration_pairs())
    runs_dir = data_dir / "experiments" / "exp2s" / "runs"
    idx = {}
    for player, pages in ((PLAYER, PAGES), (BENCHY, BENCHY_PAGES)):
        norm = normalize_name(player)
        rid = f"exp2s_{cohort}_{norm.replace(' ', '_')}"
        ledger = Ledger(runs_dir / f"{rid}.sqlite")
        ans = end_to_end(player, rid, ledger,
                         search=RoutedSearch(default=list(pages)), gate=gate,
                         llm=make_fake_llm(pages, player),
                         fetch=make_fetch(player, pages),
                         eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)
        ledger.close()
        idx[norm] = {"done": True, "run_id": rid, "stop_reason": ans.stop_reason,
                     "certificate": ans.certificate, "total": ans.total,
                     "n_records": ans.n_records}
    import json
    (runs_dir.parent / "runs_index.json").write_text(json.dumps(idx, indent=2),
                                                     encoding="utf-8")
    return cohort


@pytest.fixture
def gate():
    # fitted on 25 synthetic correct examples with confidence >= 0.8
    pairs = [(0.8 + 0.008 * i, True) for i in range(25)]
    return ConformalGate.fit(pairs)


@pytest.fixture
def fake_llm():
    return make_fake_llm()


@pytest.fixture
def fake_search():
    return FakeSearch()
