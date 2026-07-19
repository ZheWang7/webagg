"""Chapter 7 unit tests, LOCKED to the paper's worked run (guide §7.4).

The Chao value 21.0 below is the paper's worked-run T=4 row: keeping the
code and the paper numerically locked is free regression testing.
"""
import math
import pytest

from webagg.frontier import (FrontierState, Formulation, StratumState,
                             stratum_pools, w_g, add_if_novel,
                             update_yield_estimates, prune_formulations)


# ---- the silent unit bug (guide §7.3: "the most damaging porting mistake") --

def test_occasions_not_sources():
    # One formulation surfaced r1 -- even if via 4 different pages, `covered`
    # holds FORMULATION ids, so r1 stays a singleton and f1 stays honest.
    s = FrontierState(); s.T = 1
    s.covered["r1"] = {"f1"}; s.record_stratum["r1"] = "g"
    assert s.f(1, {"g"}) == 1          # still a singleton


# ---- frontier credit is a sum of yields, not a count (guide §7.4) ----------

def test_frontier_credit_uses_yields_not_count():
    s = FrontierState(); s.T = 2
    s.covered["r1"] = {"f1", "f2"}; s.record_stratum["r1"] = "g"
    for i in range(10):                # 10 junk formulations, ~0.001 rec each
        s.formulations[f"j{i}"] = Formulation(f"j{i}", "q", 0.01, 0.1)
    assert s.U_hat({"g"}) < 0.5        # junk barely moves the estimate
    s.formulations["big"] = Formulation("big", "registry sweep", 0.9, 30)
    assert s.U_hat({"g"}) > 1.0        # one big pending unlock dominates


# ---- Chao brake, bias-corrected f2=0 form (locked to worked run T=4) -------

def test_chao_bias_corrected_when_no_doubletons():
    s = FrontierState(); s.T = 4
    for i in range(8):
        s.covered[f"r{i}"] = {f"f{i}"}; s.record_stratum[f"r{i}"] = "g"
    assert s.chao_m0({"g"}) == (3 / 4) * 8 * 7 / 2      # 21.0


# ---- cardinality brake (App. E) blocks stopping ----------------------------

def test_cardinality_brake_blocks_stop():
    # claimed_count=3 but only N=2 found, with otherwise-passing stats
    # (certificate + cold frontier) -> all_strata_pass must be False.
    from webagg.pipeline import all_strata_pass
    s = FrontierState()
    s.strata["g"] = StratumState(name="g", claimed_count=3)
    s.T = 5
    s.covered["r1"] = {"f1", "f2", "f3"}; s.record_stratum["r1"] = "g"
    s.covered["r2"] = {"f1", "f2"};       s.record_stratum["r2"] = "g"
    # no singletons, no pending formulations, absurdly loose eps: the ONLY
    # thing standing between the loop and a stop is the brake
    assert s.f(1, {"g"}) == 0
    assert not all_strata_pass(s, eps_g=1e9, delta_M=0.1, eta=0.2, max_steps=10)
    s.strata["g"].claimed_count = 2       # claim satisfied -> brake releases
    assert all_strata_pass(s, eps_g=1e9, delta_M=0.1, eta=0.2, max_steps=10)


# ---- reservation index (App. B): long shots with fat upside win ------------

def test_reservation_index_prefers_long_shots():
    a = Formulation("a", "podcast", 0.3, 1, 0.02)
    b = Formulation("b", "interview", 0.05, 1, 0.02)
    assert a.reservation_index(0.5) > b.reservation_index(0.5) > 0
    assert a.reservation_index(0.0) < 0    # lam=0 -> economic stop


# ---- two-conjunct rule: a hot frontier blocks stopping (guide §7.3) --------

def test_hot_frontier_blocks_stop():
    from webagg.pipeline import all_strata_pass
    s = FrontierState()
    s.strata["g"] = StratumState(name="g")
    s.T = 3
    s.covered["r1"] = {"f1", "f2"}; s.record_stratum["r1"] = "g"   # no singletons
    # pending generic formulation promising >= eta records: conjunct (ii) fails
    s.formulations["hot"] = Formulation("hot", "registry sweep", 0.9, 5)
    assert not all_strata_pass(s, eps_g=1e9, delta_M=0.1, eta=0.2, max_steps=10)
    s.formulations["hot"].issued = True    # frontier goes cold
    assert all_strata_pass(s, eps_g=1e9, delta_M=0.1, eta=0.2, max_steps=10)


# ---- the radius is honest: at toy scale psi dwarfs eps_g -------------------

def test_psi_is_large_at_toy_scale():
    s = FrontierState()
    for i in range(5):                     # N_g = 5 records
        s.covered[f"r{i}"] = {"f0"}; s.record_stratum[f"r{i}"] = "g"
    psi = s.psi({"g"}, delta_M=0.10, w_g=1.0, max_occasions=200)
    assert psi > 0.10                      # certificate CANNOT fire statistically
    # tightening via realized V helps but must stay positive
    assert s.psi({"g"}, 0.10, 1.0, 200, V_realized=1.0) > 0


# ---- housekeeping helpers --------------------------------------------------

def test_add_if_novel_dedupes_rephrasings():
    s = FrontierState()
    assert add_if_novel(s, Formulation("x", "Acme funding round", 0.5, 2))
    assert not add_if_novel(s, Formulation("y", "funding Acme round", 0.5, 2))
    assert len(s.formulations) == 1

def test_update_yield_estimates_shrinks_but_is_idempotent():
    s = FrontierState()
    done = Formulation("d", "q1", 0.9, 10); done.issued = True
    done.realized_yield = 1                # LLM promised 9, delivered 1
    pend = Formulation("p", "q2", 0.8, 5)
    s.formulations = {"d": done, "p": pend}
    f1 = update_yield_estimates(s, done)
    assert f1 < 1.0 and pend.p_success == pytest.approx(0.8 * f1)
    f2 = update_yield_estimates(s, done)   # calling again must NOT compound
    assert pend.p_success == pytest.approx(0.8 * f2)

def test_prune_formulations_zeroes_only_target_stratum():
    s = FrontierState()
    s.formulations["a"] = Formulation("a", "q", 0.5, 2, stratum="g")
    s.formulations["b"] = Formulation("b", "q2", 0.5, 2, stratum=None)
    assert prune_formulations(s, "g") == 1
    assert s.formulations["a"].residual_yield == 0.0
    assert s.formulations["b"].residual_yield > 0      # generic survives
