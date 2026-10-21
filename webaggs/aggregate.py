"""Aggregation with the honest two-term interval (webaggr.pdf §7.3).

f_Q for Exp 2-S: SUM of adopted (player, season) points over distinct
resolved records. The reported interval is (eps_C + eps_F) * SUM where
  eps_C = the certified unseen-mass level (epsilon if statistically
          certified; U_hat + psi at stop otherwise — honest, possibly wide)
  eps_F = per-domain calibrated fidelity level. The NBA domain has NO
          calibrated eps_F yet, so the report carries eps_F=None and is
          labeled 'fidelity uncalibrated' — never a silently-invented number.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Answer:
    stratum: str
    total: int                      # f_Q = SUM of adopted season points
    n_records: int
    eps_c: float                    # completeness term actually certified
    eps_f: float | None             # None -> fidelity uncalibrated (loud)
    certificate: str                # "statistical" | "checksum" | "abandoned"
    stop_reason: str                # certified | economic | frontier_exhausted | budget
    interval: tuple | None = None   # (lo, hi) only when eps_F exists
    records: list = field(default_factory=list)
    delta_plus: float | None = None # SUM-claim missing-value bound, if any


def aggregate_with_ci(stratum, resolved_records, *, eps_c, eps_f,
                      certificate, stop_reason, delta_plus=None) -> Answer:
    total = sum(r.value for r in resolved_records)   # SUM over distinct records
    ans = Answer(stratum=stratum, total=total, n_records=len(resolved_records),
                 eps_c=eps_c, eps_f=eps_f, certificate=certificate,
                 stop_reason=stop_reason, records=list(resolved_records),
                 delta_plus=delta_plus)
    if eps_f is not None:                            # two-term interval (Cor. 1)
        half = (eps_c + eps_f) * total
        ans.interval = (total - half, total + half)
    return ans
