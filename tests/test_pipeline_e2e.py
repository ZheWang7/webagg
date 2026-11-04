"""End-to-end over the synthetic citation worlds — the full loop with
injected search/fetch/llm (tests only), then replay self-tests over the
produced ledgers, as replay_exp2c.py runs them."""
import os

from tests.conftest import (BIG_PAGES, BIG_TRUTH, BIGPAPER, PAGES, TARGET,
                            TRUTH, FakeSearch, RoutedSearch, fake_fetch,
                            make_fake_llm, make_fetch)
from webaggs.pipeline import end_to_end
from webaggs.replay import recall_at_step, replay_stop
from webaggs.storage import Ledger


def _run(tmp_path, gate, paper=TARGET, pages=PAGES, run_id="t1", **kw):
    ledger = Ledger(tmp_path / f"{run_id}.sqlite")
    defaults = dict(eps_g=0.30, delta_m=0.05, budget_usd=5.0, max_steps=30)
    defaults.update(kw)
    ans = end_to_end(paper, run_id, ledger,
                     search=RoutedSearch(default=list(pages)), gate=gate,
                     llm=make_fake_llm(pages, paper),
                     fetch=make_fetch(paper, pages), **defaults)
    return ledger, ans


def test_small_world_dedups_rejects_contamination_ends_honest(tmp_path, gate):
    """3 true citers across 3 domains; a doi-URL duplicate must collapse to
    one record; the mistargeted row must be rejected by the ER guard; with
    claims OFF and n tiny, the only honest ending is abandoned."""
    ledger, ans = _run(tmp_path, gate)
    assert ans.certificate == "abandoned"
    assert ans.n_records == 3 and ans.total == 3   # COUNT via value=1
    assert sorted(r.record_key.split("|")[1] for r in ans.records) == \
        sorted(TRUTH)
    rej = ledger.conn.execute(
        "SELECT entity_surface FROM mentions WHERE accepted=0").fetchall()
    assert ("Some Other Famous Paper",) in rej     # contamination stored, not counted
    # the duplicate (AAA.1 via doi.org URL) corroborated the same record
    dup = [r for r in ans.records if r.record_key.endswith("10.1234/aaa.1")]
    assert dup and dup[0].n_domains == 2
    ledger.close()


def test_statistical_certificate_fires_end_to_end(tmp_path, gate):
    """THE Exp 2-C path: 400 citers fully double-covered, frontier cold,
    max_steps=4 -> psi ~= 0.294 < eps 0.30 -> a STATISTICAL stop, labeled
    as such, with recall 1.0 at the stop."""
    ledger, ans = _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES,
                       run_id="big", max_steps=4)
    assert ans.stop_reason == "certified"
    assert ans.certificate == "statistical"        # not checksum: claims off
    assert ans.n_records == 400 and ans.total == 400
    live = ledger.stop("big")
    ms = ledger.measurements("big", metric="U_hat")
    rr = replay_stop(ms, 0.30)                     # S1: replay reproduces it
    assert rr.stop_step == live[0]
    truth_keys = {f"{k.lower() and _norm(BIGPAPER)}|{k}" for k in BIG_TRUTH}
    assert recall_at_step(ledger.coverage("big"), truth_keys,
                          rr.stop_step) == 1.0
    # S2: monotone in eps over the same ledger
    prev = None
    for eps in (0.30, 0.40, 0.50):
        r = replay_stop(ms, eps)
        if prev is not None:
            assert r.stop_step is not None and r.stop_step <= prev
        prev = r.stop_step if r.stop_step is not None else prev
    ledger.close()


def _norm(p):
    from webaggs.canon import normalize_name
    return normalize_name(p)


def test_ledger_has_everything_replay_needs(tmp_path, gate):
    ledger, _ = _run(tmp_path, gate, run_id="t4")
    ms = ledger.measurements("t4", metric="U_hat")
    assert ms
    for _s, _g, _m, _u, extra in ms:
        for key in ("psi", "hot", "N", "claimed_count", "certified", "spent",
                    "V_realized", "psi_realized"):
            assert key in extra
        assert extra["certified"] is not None
        assert extra["claimed_count"] is None      # claims off: brake unarmed
    assert ledger.coverage("t4")
    ledger.close()


def test_prompts_load_from_foreign_cwd(tmp_path, gate, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger, ans = _run(tmp_path, gate, run_id="t6")
    assert ans.n_records == 3
    ledger.close()
    assert os.getcwd() == str(tmp_path)


def test_fakes_route_like_the_real_thing(tmp_path, gate):
    assert FakeSearch().search("anything")          # back-compat shims work
    src = fake_fetch("https://api.openalex.org/works", "f")
    assert src.status == "denied"                   # fake honors REAL deny list
