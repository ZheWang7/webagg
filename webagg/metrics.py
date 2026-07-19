"""The measurement spine: everything that happens gets logged here."""
from .storage import MeasurementRow, FrontierFormulationRow, StratumStateRow


def log_measurement(session, run_id, step, metric, value, *,
                    stratum=None, extra=None):
    # stratum: the entity/stratum key this metric belongs to (guide §4.2).
    # Per-stratum metrics (U_hat, psi, m0, kappa, f1/f2...) must pass it;
    # run-global metrics (token cost, latency) leave it None.
    session.add(MeasurementRow(
        run_id=run_id, step=step, stratum=stratum, metric=metric,
        value=float(value), extra=extra or {},
    ))


def log_stop(session, run_id, step, *, reason, stratum=None, extra=None):
    """NEW (guide ch. 7). One row per stop event, with WHY the loop halted:
    'certified'  -- every stratum passed the two-conjunct rule (a certificate)
    'economic'   -- best reservation index <= 0 (App. B; NOT a certificate)
    'budget'     -- spend cap hit (NOT a certificate)
    'frontier_exhausted' -- no unissued formulations left (NOT a certificate)
    The report layer (§13) reads this to decide which interval to print."""
    e = {"reason": reason}
    if extra:
        e.update(extra)
    log_measurement(session, run_id, step, "stop", 1.0, stratum=stratum, extra=e)


def log_formulation(session, run_id, formulation, step=0):
    session.merge(FrontierFormulationRow(
        formulation_id=formulation.formulation_id,
        run_id=run_id, step=step,
        query=formulation.query,
        # legacy column: collapsed expectation, for old readers/demos
        expected_yield=formulation.p_success * formulation.yield_if_success,
        issued=int(formulation.issued),
        realized_yield=formulation.realized_yield,
        # NEW (ch. 7): the two-point yield model + intent provenance
        p_success=formulation.p_success,
        yield_if_success=formulation.yield_if_success,
        stratum=formulation.stratum,
        gap_directed=int(formulation.gap_directed),
    ))


def persist_stratum_states(session, run_id, state, step):
    """NEW (guide ch. 7). Snapshot every stratum's stopping state (the
    certificate's terms) into stratum_states. Called at loop exit."""
    from .frontier import stratum_pools, w_g
    from . import config
    for g, pool in stratum_pools(state):
        S = state.strata[g]
        session.add(StratumStateRow(
            run_id=run_id, stratum=g, certified=S.certified,
            claimed_count=S.claimed_count,
            n_records=state.N(pool), f1=state.f(1, pool), f2=state.f(2, pool),
            u_hat=state.U_hat(pool),
            psi=state.psi(pool, config.DELTA_M, w_g(state, g),
                          config.MAX_STEPS,
                          V_realized=S.V or None),
            step=step,
        ))
