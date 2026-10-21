"""Boundary and edge-condition tests: the ends of every parameter range."""
import pytest

from tests.conftest import RoutedSearch, make_fake_llm, make_fetch
from webaggs.aggregate import aggregate_with_ci
from webaggs.corroborate import ResolvedRecord
from webaggs.er import resolve_entity
from webaggs.oracle_nba import NBAStatsDriver, OracleError
from webaggs.pipeline import end_to_end
from webaggs.storage import Ledger
from webaggs.truth import (grade, load_display_names, load_truth, season_key,
                           write_truth)

PLAYER = "Edgar Casey"


def _world(n_seasons, claims=()):
    truth = {f"{2000 + i}-{(2001 + i) % 100:02d}": 100 + i
             for i in range(n_seasons)}
    pages = {"https://a.test/edgar": {
        "rows": [(PLAYER, s, v, 0.95) for s, v in truth.items()],
        "claims": list(claims)}}
    return truth, pages


def _run(tmp_path, gate, pages, rid="b", **kw):
    ledger = Ledger(tmp_path / f"{rid}.sqlite")
    ans = end_to_end(PLAYER, rid, ledger,
                     search=RoutedSearch(default=list(pages)), gate=gate,
                     llm=make_fake_llm(pages, PLAYER),
                     fetch=make_fetch(PLAYER, pages),
                     eps_g=0.05, delta_m=0.05, **kw)
    return ledger, ans


def test_occasion_novelty_counter_capped_at_y_cap(tmp_path, gate):
    """Assumption 1(b): the per-occasion novelty COUNTER is capped at Y_CAP
    even when one page yields 20 records (all records still enter N)."""
    truth, pages = _world(20)
    ledger, ans = _run(tmp_path, gate, pages, budget_usd=5.0, max_steps=10)
    assert ans.n_records == 20
    news = [e["new"] for _s, _g, _m, _u, e in ledger.measurements("b", "U_hat")]
    assert max(news) == 12                    # Y_CAP, not 20
    ledger.close()


def test_budget_stop_fires_before_certification(tmp_path, gate):
    truth, pages = _world(3, claims=[{"kind": "COUNT", "value": 3}])
    # one search costs 0.02 > budget 0.01 -> budget stop at step 1 — wait:
    # spend is checked AFTER the occasion, so step 1 runs, then stops.
    ledger, ans = _run(tmp_path, gate, pages, rid="budget",
                       budget_usd=0.01, max_steps=10)
    stop = ledger.stop("budget")
    # checksum may certify in the same step; either label is honest, but the
    # run must end at step 1 and never claim a STATISTICAL certificate
    assert stop[0] == 1 and ans.certificate != "statistical"
    ledger.close()


def test_max_steps_exhaustion_is_labeled_honestly(tmp_path, gate):
    truth, pages = _world(3)                 # no claims: cannot certify
    ledger, ans = _run(tmp_path, gate, pages, rid="cap",
                       budget_usd=5.0, max_steps=1)
    assert ledger.stop("cap")[1] == "max_steps"
    assert ans.certificate == "abandoned"
    ledger.close()


def test_economic_stop_when_only_junk_formulations_remain(tmp_path, gate):
    truth, pages = _world(2)
    base = make_fake_llm(pages, PLAYER)
    step = {"n": 0}

    def junky_llm(system, user, **kw):       # followups propose only junk
        if kw.get("purpose") == "followups":
            step["n"] += 1
            return {"formulations": [{"query": f"junk {step['n']}",
                                      "p_success": 0.05,
                                      "yield_if_success": 0.1}]}
        return base(system, user, **kw)

    ledger = Ledger(tmp_path / "econ.sqlite")
    end_to_end(PLAYER, "econ", ledger,
               search=RoutedSearch(default=list(pages)), gate=gate,
               llm=junky_llm, fetch=make_fetch(PLAYER, pages),
               eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)
    assert ledger.stop("econ")[1] == "economic"   # junk not worth its cost
    ledger.close()


def test_irrelevant_page_feeds_the_phi_audit(tmp_path, gate):
    truth, pages = _world(2)
    pages["https://soup.test/recipes"] = {"rows": [], "claims": []}

    def fetch(url, fid):                     # the soup page never names Edgar
        src = make_fetch(PLAYER, pages)(url, fid)
        if "soup" in url:
            src.main_text = "a long page about minestrone " * 30
        return src

    ledger = Ledger(tmp_path / "phi.sqlite")
    end_to_end(PLAYER, "phi", ledger, search=RoutedSearch(default=list(pages)),
               gate=gate, llm=make_fake_llm(pages, PLAYER), fetch=fetch,
               eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=10)
    rej = ledger.conn.execute("SELECT url FROM rejected_sources").fetchall()
    assert ("https://soup.test/recipes",) in rej
    ledger.close()


def test_empty_search_results_are_a_zero_yield_occasion(tmp_path, gate):
    truth, pages = _world(2)
    ledger, ans = _run(tmp_path, gate, {}, rid="empty",
                       budget_usd=5.0, max_steps=5)
    # nothing found: U_hat pinned at 1.0, run ends without a certificate
    ms = ledger.measurements("empty", "U_hat")
    assert ms and all(u == 1.0 for _s, _g, _m, u, _e in ms)
    assert ans.certificate == "abandoned" and ans.n_records == 0
    ledger.close()


# --- truth / grading / oracle boundaries ------------------------------------
def test_truth_refuses_duplicate_rows_on_write_and_load(tmp_path):
    rows = [{"player": "A B", "player_id": 1, "season": "2019-20",
             "pts": 1, "gp": 1}] * 2
    with pytest.raises(RuntimeError, match="Duplicate"):
        write_truth(tmp_path, "dup", rows)
    # and a corrupt file smuggled past write_truth is caught at load
    ok = [{"player": "A B", "player_id": 1, "season": "2019-20",
           "pts": 1, "gp": 1}]
    csv_path = write_truth(tmp_path, "dup2", ok)
    text = csv_path.read_text()
    csv_path.write_text(text + text.splitlines()[1] + "\n")
    import hashlib
    import json
    man = csv_path.parent / "manifest.json"
    m = json.loads(man.read_text())
    m["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    man.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="duplicate"):
        load_truth(tmp_path, "dup2")


def test_display_names_roundtrip(tmp_path):
    write_truth(tmp_path, "names", [{"player": "Luka Dončić", "player_id": 1,
                                     "season": "2019-20", "pts": 1, "gp": 1}])
    assert load_display_names(tmp_path, "names") == \
        {"luka doncic": "Luka Dončić"}


def test_season_key_rejects_garbage():
    with pytest.raises(ValueError):
        season_key("Career")


def test_grade_with_zero_truth_player():
    g = grade("p", {}, {"2019-20": 5})
    assert g.record_recall == 0.0 and g.n_spurious == 1


def test_aggregate_interval_when_eps_f_exists():
    recs = [ResolvedRecord("p|2019-20", "2019-20", 1000, 1.0, 2)]
    ans = aggregate_with_ci("p", recs, eps_c=0.05, eps_f=0.10,
                            certificate="statistical", stop_reason="certified")
    assert ans.interval == (1000 - 150, 1000 + 150)   # (eps_C+eps_F)*SUM


def test_er_empty_surface_is_rejected():
    assert resolve_entity("", "Jalen Brunson") is None
    assert resolve_entity("  . ", "Jalen Brunson") is None


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload

    def json(self):
        return self._p


class _Client:
    def __init__(self, payloads):
        self.payloads = [_Resp(p) for p in payloads]

    def get(self, url, params=None, headers=None, timeout=None):
        return self.payloads.pop(0)


def _players_payload(rows):
    return {"resultSets": [{"headers": ["PERSON_ID", "DISPLAY_FIRST_LAST"],
                            "rowSet": rows}]}


def test_oracle_player_lookup_exact_zero_and_ambiguous():
    d = NBAStatsDriver(pace_seconds=0, client=_Client([_players_payload(
        [[1, "Jalen Brunson"], [2, "Jalen Johnson"], [3, "Jalen Johnson"]])]))
    assert d.find_player_id("Jalen Brunson") == (1, "Jalen Brunson")
    d = NBAStatsDriver(pace_seconds=0, client=_Client([_players_payload([])]))
    with pytest.raises(OracleError, match="0 matches"):
        d.find_player_id("Nobody Atall")
    d = NBAStatsDriver(pace_seconds=0, client=_Client([_players_payload(
        [[2, "Jalen Johnson"], [3, "Jalen Johnson"]])]))
    with pytest.raises(OracleError, match="2 matches"):
        d.find_player_id("Jalen Johnson")


def test_oracle_zero_season_rows_raise():
    payload = {"resultSets": [{"name": "SeasonTotalsRegularSeason",
                               "headers": ["SEASON_ID", "TEAM_ABBREVIATION",
                                           "PTS", "GP"],
                               "rowSet": []}]}
    d = NBAStatsDriver(pace_seconds=0, client=_Client([payload]))
    with pytest.raises(OracleError, match="zero regular-season"):
        d.career_regular_season(1)
