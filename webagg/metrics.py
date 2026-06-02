"""The measurement spine: everything that happens gets logged here."""
from .storage import MeasurementRow, FrontierFormulationRow


def log_measurement(session, run_id, step, metric, value, *, extra=None):
    session.add(MeasurementRow(
        run_id=run_id, step=step, metric=metric,
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
