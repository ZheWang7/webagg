from webaggs.replay import recall_at_step, replay_stop, stop_predicate


def _row(step, u, psi, hot=False, n=10, claimed=None, cert=""):
    return (step, "p", "U_hat", u,
            {"psi": psi, "hot": hot, "N": n, "claimed_count": claimed,
             "certified": cert})


def test_predicate_two_conjuncts_and_brake():
    assert stop_predicate(0.01, 0.02, hot=False, n=10, claimed_count=None,
                          certified="", eps=0.05)
    assert not stop_predicate(0.01, 0.02, hot=True, n=10, claimed_count=None,
                              certified="", eps=0.05)       # hot frontier blocks
    assert not stop_predicate(0.01, 0.05, hot=False, n=10, claimed_count=None,
                              certified="", eps=0.05)       # u+psi not < eps
    assert not stop_predicate(0.0, 0.0, hot=False, n=2, claimed_count=3,
                              certified="", eps=0.5)        # cardinality brake


def test_predicate_is_nan_proof_on_certified_column():
    """The exp3/exp7 lesson: '' must mean NOT certified; only a non-empty
    string passes. (float('nan') would be truthy — the bug this guards.)"""
    assert not stop_predicate(1.0, 1.0, False, 1, None, "", eps=0.05)
    assert stop_predicate(1.0, 1.0, False, 1, None, "checksum", eps=0.05)
    assert not stop_predicate(1.0, 1.0, False, 1, None, str(float("nan")) and "",
                              eps=0.05)


def test_replay_finds_first_passing_step_and_censoring():
    rows = [_row(1, 0.5, 0.3), _row(2, 0.2, 0.2), _row(3, 0.05, 0.04)]
    rr = replay_stop(rows, eps=0.10)
    assert rr.stop_step == 3 and not rr.censored
    rr = replay_stop(rows, eps=0.45)                 # 0.2+0.2 < 0.45 at step 2
    assert rr.stop_step == 2
    rr = replay_stop(rows, eps=0.01)
    assert rr.stop_step is None and rr.censored      # never fires: censored


def test_recall_at_step_counts_only_truth_records():
    cov = [(1, "p|2019-20"), (2, "p|bogus"), (3, "p|2020-21")]
    truth = {"p|2019-20", "p|2020-21", "p|2021-22"}
    assert abs(recall_at_step(cov, truth, 2) - 1 / 3) < 1e-9
    assert abs(recall_at_step(cov, truth, None) - 2 / 3) < 1e-9


def test_recall_trajectory_is_cumulative_per_event():
    cov = [(1, "p|2019-20"), (1, "p|junk"), (3, "p|2020-21")]
    truth = {"p|2019-20", "p|2020-21", "p|2021-22"}
    from webaggs.replay import recall_trajectory
    traj = recall_trajectory(cov, truth)
    assert traj == [(1, 1 / 3), (1, 1 / 3), (3, 2 / 3)]
    assert dict(traj) == {1: 1 / 3, 3: 2 / 3}      # last-per-step view
