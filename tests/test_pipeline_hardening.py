"""Adversarial end-to-end: paywalls, oracle denial inside the loop, cross-
player contamination, duplicate URLs across formulations, and the full
claims-engine loop (brake -> gap-directed search -> COUNT closure)."""
from tests.conftest import RoutedSearch, make_fake_llm, make_fetch
from webaggs.pipeline import end_to_end
from webaggs.storage import Ledger

PLAYER = "Benchy Marks"
TRUTH = {"2015-16": 200, "2016-17": 350, "2018-19": 120, "2019-20": 90}  # 760

PAGES = {
    # the withheld oracle shows up in search results: must be DENIED in-loop
    "https://stats.nba.com/benchy": {"rows": [], "claims": []},
    # a subscription site: censored observation, logged, no vote
    "https://paywall.test/benchy": {"rows": [], "claims": [],
                                    "status": "paywall"},
    # two ordinary pages; both state the COUNT=4 claim; page A also carries a
    # contaminating row about a DIFFERENT player sharing the surname
    "https://a.test/benchy": {
        "rows": [("Benchy Marks", "2015-16", 200, 0.92),
                 ("Benchy Marks", "2016-17", 350, 0.92),
                 ("Corky Marks", "2016-17", 999, 0.95)],   # contamination
        "claims": [{"kind": "COUNT", "value": 4}]},
    "https://b.test/benchy": {
        "rows": [("B. Marks", "2018-19", 120, 0.9)],        # name variant: merges
        "claims": [{"kind": "COUNT", "value": 4}]},
    # reachable ONLY via a gap-directed query (claims-engine §5.2 behavior)
    "https://c.test/benchy-late": {
        "rows": [("Benchy Marks", "2019-20", 90, 0.9)],
        "claims": []},
}

SEARCH = RoutedSearch(
    routes=[("rookie", ["https://c.test/benchy-late"]),
            ("early career", ["https://c.test/benchy-late"])],
    default=["https://stats.nba.com/benchy", "https://paywall.test/benchy",
             "https://a.test/benchy", "https://b.test/benchy"])


def test_brake_gap_direction_closure_and_guards(tmp_path, gate):
    ledger = Ledger(tmp_path / "hard.sqlite")
    ans = end_to_end(PLAYER, "hard", ledger, search=SEARCH, gate=gate,
                     llm=make_fake_llm(PAGES, PLAYER),
                     fetch=make_fetch(PLAYER, PAGES),
                     eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)

    # 1. the COUNT=4 claim (two domains) braked stopping at N=3, emitted
    #    gap-directed searches, page C was found, and closure certified
    assert ans.certificate == "checksum" and ans.stop_reason == "certified"
    assert ans.n_records == 4
    assert ans.total == sum(TRUTH.values())                 # 760 exactly

    # 2. the oracle URL was denied and the paywall stub classified as censored
    outcomes = {status for (_r, _s, _u, _d, status, _l) in ledger.conn.execute(
        "SELECT * FROM fetch_outcomes").fetchall()}
    assert "denied" in outcomes and "paywall" in outcomes

    # 3. the contamination guard rejected Corky Marks (stored, not counted)
    rej = ledger.conn.execute(
        "SELECT entity_surface FROM mentions WHERE accepted=0").fetchall()
    assert ("Corky Marks",) in rej
    assert all("corky" not in r.record_key for r in ans.records)

    # 4. the 'B. Marks' variant merged onto the player (deterministic ER)
    assert any(r.season == "2018-19" and r.value == 120 for r in ans.records)
    ledger.close()
