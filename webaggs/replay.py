"""Offline stop replay from a frozen ledger (the Exp 2 design).

Valid because epsilon is read ONLY in the stop test: every quantity the test
needs at step t (U_hat, psi, hot, N, claimed_count, certified) was logged at
step t, so re-running the predicate at a looser epsilon over the recorded
trajectory reproduces exactly what a live run at that epsilon would have
decided — up to the step where the live run actually ended.

NaN discipline (the exp3/exp7 lesson): 'certified' is stored as "" or a
checksum tag, never None; replay reads it with an explicit truthiness-on-
string check and never routes it through a pandas object column.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReplayResult:
    eps: float
    stop_step: int | None        # step the certificate would fire; None = never
    live_end_step: int           # where the recorded trajectory actually ends
    censored: bool               # True if no certified stop within the recording


def stop_predicate(u: float, psi: float, hot: bool, n: int,
                   claimed_count, certified: str, eps: float) -> bool:
    """Single-stratum form of all_strata_pass (webaggr.pdf §3.3 Def. 6)."""
    if isinstance(certified, str) and certified:
        return True                               # checksum-closed: passes
    if claimed_count is not None and n < int(claimed_count):
        return False                              # cardinality hard brake
    return (u + psi < eps) and (not hot)          # the two conjuncts


def replay_stop(measurement_rows, eps: float) -> ReplayResult:
    """measurement_rows: Ledger.measurements(run_id, 'U_hat') for ONE stratum,
    ordered by step. Returns the first step the stop test passes at `eps`."""
    last_step = 0
    for step, _stratum, _metric, u, extra in measurement_rows:
        last_step = step
        ok = stop_predicate(
            u=float(u), psi=float(extra["psi"]), hot=bool(extra["hot"]),
            n=int(extra["N"]), claimed_count=extra.get("claimed_count"),
            certified=str(extra.get("certified") or ""), eps=eps)
        if ok:
            return ReplayResult(eps=eps, stop_step=step,
                                live_end_step=last_step, censored=False)
    return ReplayResult(eps=eps, stop_step=None,
                        live_end_step=last_step, censored=True)


def recall_trajectory(coverage_rows, truth_keys: set[str]) -> list[tuple[int, float]]:
    """(step, realized recall) after each step with a discovery — the oracle
    side of the calibration plot (Exp 2 Plot 1: predicted vs realized)."""
    found: set[str] = set()
    traj = []
    for step, rk in coverage_rows:
        if rk in truth_keys:
            found.add(rk)
        traj.append((step, len(found) / max(len(truth_keys), 1)))
    return traj


def recall_at_step(coverage_rows, truth_keys: set[str], step: int | None) -> float:
    """Realized recall at a (replayed) stopping step; step=None -> full run."""
    found = {rk for s, rk in coverage_rows
             if rk in truth_keys and (step is None or s <= step)}
    return len(found) / max(len(truth_keys), 1)
