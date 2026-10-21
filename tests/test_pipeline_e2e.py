"""End-to-end over the synthetic world — the full loop with injected
search/fetch/llm (tests only), then the replay self-tests over the ledger
it produced, exactly as replay_exp2s.py will run them."""
import os

from tests.conftest import PAGES, PLAYER, TRUTH, FakeSearch, fake_fetch, \
    make_fake_llm
from webaggs.pipeline import end_to_end, rekey_and_revalidate
from webaggs.replay import recall_at_step, replay_stop
from webaggs.storage import Ledger


def _run(tmp_path, gate, pages=PAGES, run_id="t1"):
    ledger = Ledger(tmp_path / f"{run_id}.sqlite")
    ans = end_to_end(PLAYER, run_id, ledger, search=FakeSearch(), gate=gate,
                     llm=make_fake_llm(pages), fetch=fake_fetch,
                     eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)
    return ledger, ans


def test_checksum_certified_run_assembles_the_exact_sum(tmp_path, gate):
    """Pages carry a COUNT=3 claim on two domains -> exact closure (Thm 4)
    certifies the stratum; stop reason 'certified', path 'checksum', and the
    assembled SUM equals the synthetic truth."""
    ledger, ans = _run(tmp_path, gate)
    assert ans.stop_reason == "certified"
    assert ans.certificate == "checksum"      # reported separately by prereg
    assert ans.n_records == 3
    assert ans.total == sum(TRUTH.values())   # 1450: the aggregation is right
    assert ans.eps_f is None                  # fidelity honestly uncalibrated
    assert ans.interval is None
    ledger.close()


def test_abandoned_run_reports_honest_eps_c(tmp_path, gate):
    """No aggregate claims anywhere -> no checksum path; at toy N the honest
    psi dwarfs eps (the guide's intended small-strata behavior), so the run
    ends economic/frontier_exhausted with certificate 'abandoned' and a
    truthfully wide eps_c — never a fake certificate."""
    pages = {u: {"rows": p["rows"], "claims": []} for u, p in PAGES.items()}
    ledger, ans = _run(tmp_path, gate, pages=pages, run_id="t2")
    assert ans.stop_reason in ("economic", "frontier_exhausted", "budget",
                               "max_steps")
    assert ans.certificate == "abandoned"
    assert ans.eps_c > 0.05                   # honest, not the promised level
    assert ans.total == sum(TRUTH.values())   # the answer itself is still right
    ledger.close()


def test_replay_self_tests_S1_S2_over_the_ledger(tmp_path, gate):
    """S1: replay at the live eps reproduces the live certified stop.
    S2: replayed stops are monotone in eps. Same checks replay_exp2s.py runs."""
    ledger, ans = _run(tmp_path, gate, run_id="t3")
    ms = ledger.measurements("t3", metric="U_hat")
    cov = ledger.coverage("t3")
    live = ledger.stop("t3")
    assert live is not None and live[1] == "certified"
    rr_live = replay_stop(ms, 0.05)
    assert rr_live.stop_step == live[0]                        # S1
    prev = None
    for eps in (0.05, 0.10, 0.20, 0.30):
        rr = replay_stop(ms, eps)
        if prev is not None:
            assert rr.stop_step is not None and rr.stop_step <= prev   # S2
        prev = rr.stop_step
    truth_keys = {f"testo sampleson|{s}" for s in TRUTH}
    assert recall_at_step(cov, truth_keys, rr_live.stop_step) == 1.0
    ledger.close()


def test_ledger_has_everything_replay_needs(tmp_path, gate):
    ledger, _ = _run(tmp_path, gate, run_id="t4")
    ms = ledger.measurements("t4", metric="U_hat")
    assert ms, "no measurement spine written"
    for _step, _g, _m, _u, extra in ms:
        for key in ("psi", "hot", "N", "claimed_count", "certified", "spent"):
            assert key in extra
        assert extra["certified"] is not None   # '' not None (NaN discipline)
    assert ledger.coverage("t4"), "no coverage events written"
    ledger.close()


def test_post_er_second_pass_revokes_bad_count_certificate(tmp_path, gate):
    """Impl guide pitfall #3: a COUNT certificate granted on pre-ER counts is
    revoked once the settled N disagrees. Simulated by shrinking the record
    set after certification (what an ER merge does)."""
    ledger, _ = _run(tmp_path, gate, run_id="t5")
    from webaggs.claims import ClaimsEngine
    from webaggs.frontier import FrontierState, StratumState
    from webaggs.types import Claim
    st = FrontierState()
    st.strata["p"] = StratumState(name="p", certified="checksum")
    st.covered = {"p|2019-20": {"a"}, "p|2020-21": {"a"}}   # ER merged 3 -> 2
    st.record_stratum = {k: "p" for k in st.covered}
    ce = ClaimsEngine()
    for d in ("a.test", "b.test"):
        ce.ingest(Claim(f"c{d}", "s", d, "p", "COUNT", 3))
    revoked = rekey_and_revalidate(st, ce, {}, {}, ledger, "t5")
    assert revoked == ["p"]
    assert st.strata["p"].certified == ""                    # '' not None
    ledger.close()


def test_prompts_load_from_foreign_cwd(tmp_path, gate, monkeypatch):
    """The CWD-relative prompt trap: the pipeline must run from ANY cwd."""
    monkeypatch.chdir(tmp_path)                 # a directory with no prompts/
    ledger, ans = _run(tmp_path, gate, run_id="t6")
    assert ans.n_records == 3
    ledger.close()
    assert os.getcwd() == str(tmp_path)
