import math

from webaggs.frontier import (Formulation, FrontierState, StratumState,
                              all_strata_pass)


def _state_with(records: dict[str, set], stratum="p"):
    st = FrontierState()
    st.strata[stratum] = StratumState(name=stratum)
    for rk, occasions in records.items():
        st.covered[rk] = set(occasions)
        st.record_stratum[rk] = stratum
    return st


def test_occasions_not_sources():
    """Pitfall #1: covered must hold FORMULATION ids. A record surfaced by
    two sources under ONE formulation is still a singleton (f1 counts it)."""
    st = _state_with({"p|2019-20": {"fm1"}})      # one occasion, two sources
    assert st.f(1, {"p"}) == 1                     # still a singleton
    st.covered["p|2019-20"].add("fm2")             # a SECOND formulation hits it
    assert st.f(1, {"p"}) == 0
    assert st.f(2, {"p"}) == 1


def test_u_hat_formula_matches_eq1():
    # 4 records: 2 singletons; one unissued formulation with residual 1.5
    st = _state_with({f"p|s{i}": {"a"} if i < 2 else {"a", "b"}
                      for i in range(4)})
    st.formulations["q"] = Formulation("q", "x", p_success=0.5,
                                       yield_if_success=3.0)
    # U_hat = f1/N + beta * sum(residual)/N = 2/4 + 1.0 * 1.5/4
    assert math.isclose(st.U_hat({"p"}), 0.5 + 1.5 / 4)


def test_u_hat_is_one_when_nothing_found():
    st = _state_with({})
    assert st.U_hat({"p"}) == 1.0


def test_psi_matches_guide_formula_and_shrinks_in_n():
    st = _state_with({f"p|s{i}": {"a", "b"} for i in range(10)})
    n = 10
    c = st.Y_CAP * (1 + st.BETA) / n
    expected = math.sqrt(2 * (50 * c * c) * math.log(1 / (0.05 * 1.0)))
    assert math.isclose(st.psi({"p"}, 0.05, 1.0, 50), expected)
    big = _state_with({f"p|s{i}": {"a", "b"} for i in range(100)})
    assert big.psi({"p"}, 0.05, 1.0, 50) < st.psi({"p"}, 0.05, 1.0, 50)


def test_stop_requires_both_conjuncts():
    """Def. 6: (U_hat + psi < eps) AND frontier not hot — only the
    conjunction is safe (webaggr.pdf §3.3)."""
    # Large pool, all records doubly covered -> f1 = 0, psi small
    st = _state_with({f"p|s{i}": {"a", "b"} for i in range(60)})
    st.Y_CAP = 1                                    # tight realized novelty cap
    assert all_strata_pass(st, eps_g=0.30, delta_m=0.10, eta=0.20, max_steps=4)
    # a hot pending formulation alone forbids stopping
    st.formulations["hot"] = Formulation("hot", "x", p_success=0.9,
                                         yield_if_success=2.0)
    assert not all_strata_pass(st, 0.30, 0.10, 0.20, 4)
    # and a tighter eps alone forbids stopping too
    st.formulations.clear()
    assert not all_strata_pass(st, 0.001, 0.10, 0.20, 4)


def test_cardinality_hard_brake():
    """App. E: a corroborated COUNT above N forbids stopping outright."""
    st = _state_with({f"p|s{i}": {"a", "b"} for i in range(60)})
    st.Y_CAP = 1
    st.strata["p"].claimed_count = 61               # one record still missing
    assert not all_strata_pass(st, 0.30, 0.10, 0.20, 4)
    st.strata["p"].claimed_count = 60               # closure reached
    assert all_strata_pass(st, 0.30, 0.10, 0.20, 4)


def test_certified_stratum_is_skipped():
    st = _state_with({"p|s1": {"a"}})               # f1=1 -> U_hat high
    st.strata["p"].certified = "checksum"           # stratum left the pool
    assert all_strata_pass(st, 0.05, 0.10, 0.20, 4)


def test_production_psi_identity_and_realized_variant():
    """The ported production psi: (a) with V_realized None/0 it is BIT-
    IDENTICAL to the conservative form the stop test and replay use — the
    port cannot have changed any stop decision; (b) with a realized V it is
    exactly sqrt(2*V*log(1/(delta*w)))."""
    import random
    rng = random.Random(3)
    for _ in range(100):
        st = _state_with({f"p|s{i}": {"a"} for i in range(rng.randint(0, 40))})
        st.Y_CAP = rng.choice([1, 4, 12])
        args = (rng.uniform(0.01, 0.2), rng.uniform(0.05, 1.0),
                rng.randint(1, 300))
        base = st.psi({"p"}, *args)
        assert st.psi({"p"}, *args, V_realized=None) == base
        assert st.psi({"p"}, *args, V_realized=0.0) == base   # quirk kept
        V = rng.uniform(0.01, 50)
        expect = math.sqrt(2 * V * math.log(1 / (args[0] * args[1])))
        assert math.isclose(st.psi({"p"}, *args, V_realized=V), expect)


def test_pre_occasion_c_matches_increment_bound():
    st = _state_with({f"p|s{i}": {"a"} for i in range(8)})
    assert math.isclose(st.pre_occasion_c({"p"}),
                        st.Y_CAP * (1 + st.BETA) / 8)
    empty = _state_with({})
    assert math.isclose(empty.pre_occasion_c({"p"}),
                        empty.Y_CAP * (1 + empty.BETA))      # n floored at 1
