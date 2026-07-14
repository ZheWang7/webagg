"""The measurement spine: everything that happens gets logged here."""
from .storage import MeasurementRow, FrontierFormulationRow


def log_measurement(session, run_id, step, metric, value, *,
                    stratum=None, extra=None):
    # stratum: the entity/stratum key this metric belongs to (guide §4.2).
    # Per-stratum metrics (U_hat, psi, m0, kappa, f1/f2...) must pass it;
    # run-global metrics (token cost, latency) leave it None.
    session.add(MeasurementRow(
        run_id=run_id, step=step, stratum=stratum, metric=metric,
        value=float(value), extra=extra or {},
    ))


def log_formulation(session, run_id, formulation, step=0):
    session.merge(FrontierFormulationRow(
        formulation_id=formulation.formulation_id,
        run_id=run_id, step=step,
        query=formulation.query,
        expected_yield=formulation.expected_yield,
        issued=int(formulation.issued),
        realized_yield=formulation.realized_yield,
    ))
