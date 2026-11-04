"""Adversarial end-to-end: the oracle appearing in search results, paywalls,
contamination, duplicate URL re-crediting, and gate-rejection logging."""
from tests.conftest import TARGET, RoutedSearch, make_fake_llm, make_fetch
from webaggs.pipeline import end_to_end
from webaggs.storage import Ledger

PAGES = {
    "https://api.openalex.org/works?filter=cites:W1": {"citations": []},
    "https://paywall.test/citers": {"citations": [], "status": "paywall"},
    "https://a.test/citers": {
        "citations": [
            dict(title="True Citing Paper One", doi="10.1234/c.1",
                 cited_paper=TARGET, confidence=0.95),
            dict(title="Mistargeted List Entry", doi="10.1234/zz.9",
                 cited_paper="A Different Paper Entirely", confidence=0.95),
            dict(title="Hesitant But Real Citer", doi="",
                 cited_paper=TARGET, confidence=0.30),   # below gate
        ]},
    "https://b.test/citers": {
        "citations": [
            dict(title="True Citing Paper One", doi="doi.org/10.1234/C.1",
                 cited_paper=TARGET, confidence=0.9),    # same record, 2nd domain
            dict(title="True Citing Paper Two Here", doi="",
                 cited_paper=TARGET, confidence=0.9),
        ]},
}

SEARCH = RoutedSearch(default=list(PAGES))


def test_denied_paywalled_contaminated_and_gated(tmp_path, gate):
    ledger = Ledger(tmp_path / "hard.sqlite")
    ans = end_to_end(TARGET, "hard", ledger, search=SEARCH, gate=gate,
                     llm=make_fake_llm(PAGES, TARGET),
                     fetch=make_fetch(TARGET, PAGES),
                     eps_g=0.30, delta_m=0.05, budget_usd=5.0, max_steps=20)
    # oracle URL denied in-loop; paywall logged as censored
    outcomes = {s for (_r, _st, _u, _d, s, _l) in ledger.conn.execute(
        "SELECT * FROM fetch_outcomes").fetchall()}
    assert "denied" in outcomes and "paywall" in outcomes
    # two real records; dup collapsed across domains; contamination out
    assert ans.n_records == 2 and ans.total == 2
    assert all("zz.9" not in r.record_key for r in ans.records)
    one = [r for r in ans.records if r.record_key.endswith("10.1234/c.1")][0]
    assert one.n_domains == 2
    # conformal rejection left its delta_E audit row
    gm = ledger.measurements("hard", metric="gate_rejected")
    assert gm and sum(v for _s, _g, _m, v, _e in gm) >= 1
    ledger.close()
