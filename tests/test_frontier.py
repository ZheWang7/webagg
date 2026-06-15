from webagg.frontier import FrontierState, Formulation


def test_U_hat_initially_one():
    s = FrontierState()
    assert s.U_hat() == 1.0


def test_singletons_count_distinct_records():
    s = FrontierState()
    s.covered_records["r1"] = {"src_a"}
    s.covered_records["r2"] = {"src_a", "src_b"}   # not a singleton
    s.N = 2
    assert s.singletons() == 1
    assert 0 < s.U_hat() <= 1


def test_stopping_requires_both_conditions():
    # Small singleton rate but a hot frontier => U_hat stays nonzero.
    s = FrontierState(N=20)
    for i in range(20):
        s.covered_records[f"r{i}"] = {f"src{i}", f"src{i}b"}   # 0 singletons
    s.formulations["f"] = Formulation("f", "hot query", expected_yield=3.0)
    assert s.U_hat() > 0   # frontier credit keeps it nonzero
