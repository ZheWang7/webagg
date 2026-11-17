"""End-to-end over the synthetic citation worlds — the full loop with
injected search/fetch/llm (tests only), then replay self-tests over the
produced ledgers, as replay_exp2c.py runs them."""
import os

from tests.conftest import (BIG_EPS, BIG_N, BIG_PAGE, BIG_PAGES, BIG_TRUTH,
                            BIGPAPER, PAGES, TARGET, TRUTH, FakeIndex,
                            FakeSearch, RoutedSearch, big_kwargs, fake_fetch,
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
    """THE Exp 2-C path (prereg §6.2): two paginated indexes walked one
    page per capture occasion, cursor-style; every record doubly covered;
    psi ~= 0.49 < eps 0.5 once both walks finish -> a STATISTICAL stop,
    labeled as such, with recall 1.0 at the stop."""
    kw = big_kwargs()
    ledger, ans = _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES,
                       run_id="big", budget_usd=50.0, **kw)
    assert ans.stop_reason == "certified"
    assert ans.certificate == "statistical"        # not checksum: claims off
    assert ans.n_records == BIG_N and ans.total == BIG_N
    # one occasion per page: no step ever credited more than the page size
    ms = ledger.measurements("big", metric="U_hat")
    assert max(e["new"] for *_x, e in ms) <= BIG_PAGE
    assert not ledger.measurements("big", metric="intake_truncated")
    # each index was walked exactly once, page by page
    assert [ix.calls for ix in kw["indexes"]] == [BIG_N // BIG_PAGE] * 2
    live = ledger.stop("big")
    rr = replay_stop(ms, BIG_EPS)                  # S1: replay reproduces it
    assert rr.stop_step == live[0]
    truth_keys = {f"{_norm(BIGPAPER)}|{k}" for k in BIG_TRUTH}
    assert recall_at_step(ledger.coverage("big"), truth_keys,
                          rr.stop_step) == 1.0
    # the certificate could NOT have fired before the walks finished:
    # the frontier stays hot while a next page is pending (conjunct ii)
    assert live[0] > 2 * (BIG_N // BIG_PAGE)
    # S2: monotone in eps over the same ledger
    prev = None
    for eps in (0.5, 0.7, 0.9):
        r = replay_stop(ms, eps)
        if prev is not None:
            assert r.stop_step is not None and r.stop_step <= prev
        prev = r.stop_step if r.stop_step is not None else prev
    # the index pages are ordinary ledger sources of class citation_index
    n_idx = ledger.conn.execute(
        "SELECT COUNT(*) FROM sources WHERE source_class='citation_index' "
        "AND status='ok'").fetchone()[0]
    assert n_idx == 2 * (BIG_N // BIG_PAGE)
    ledger.close()


def test_partial_overlap_keeps_singletons_and_refuses_to_certify(tmp_path, gate):
    """Honesty check: if index B misses 30% of index A, those records stay
    singletons, U_hat ~= 0.3, and U_hat + psi > eps — the run must END
    WITHOUT a certificate even though every truth record was found."""
    ledger, ans = _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES,
                       run_id="part", budget_usd=50.0, **big_kwargs(overlap=0.7))
    assert ans.n_records == BIG_N                  # all found ...
    assert ans.certificate == "abandoned"          # ... but not certifiable
    last = ledger.measurements("part", metric="U_hat")[-1]
    assert 0.25 < last[3] < 0.35                   # U_hat = singleton share
    ledger.close()


def test_failed_index_page_is_retried_not_silently_skipped(tmp_path, gate):
    """A 503 on one page must not end that index's walk: the page is
    re-enqueued (bounded) and the certificate still fires."""
    kw = big_kwargs()
    kw["indexes"][0].fail_once = {100}             # cursor offset 100 fails once
    ledger, ans = _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES,
                       run_id="retry", budget_usd=50.0, **kw)
    assert ans.n_records == BIG_N and ans.certificate == "statistical"
    errs = ledger.conn.execute("SELECT COUNT(*) FROM fetch_outcomes "
                               "WHERE status='error'").fetchone()[0]
    assert errs == 1
    ledger.close()


def test_oracle_can_never_be_walked_as_an_index(tmp_path, gate):
    """Even if someone wires the oracle's domain in as an 'index', every
    page comes back DENIED and contributes nothing."""
    kw = big_kwargs()
    kw["indexes"] = [FakeIndex("oracle-as-index",
                               [v["doi"] for v in BIG_TRUTH.values()],
                               domain="api.opencitations.net")]
    ledger, ans = _run(tmp_path, gate, paper=BIGPAPER, pages={},
                       run_id="deny", budget_usd=50.0, **kw)
    assert kw["indexes"][0].calls == 0             # never even requested
    denied = ledger.conn.execute("SELECT COUNT(*) FROM fetch_outcomes "
                                 "WHERE status='denied'").fetchone()[0]
    assert denied >= 1 and ans.certificate == "abandoned"
    ledger.close()


def test_index_walk_guards(tmp_path, gate):
    import pytest
    with pytest.raises(ValueError, match="exceeds Y_CAP"):
        _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES, run_id="g1",
             **big_kwargs(page_size=50, y_cap=10))
    with pytest.raises(ValueError, match="target and a cutoff"):
        _run(tmp_path, gate, paper=BIGPAPER, pages=BIG_PAGES, run_id="g2",
             **big_kwargs(cutoff=None))


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
    src = fake_fetch("https://api.opencitations.net/index/v2/citations/doi:10.1/x", "f")
    assert src.status == "denied"                   # fake honors REAL deny list
