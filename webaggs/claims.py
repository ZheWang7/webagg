"""Claims engine: aggregate claims as checksums (webaggr.pdf §5.2, Thm 4).

NBA pages state exactly the two claim kinds the theory wants:
  COUNT — "played N NBA seasons"      -> exact closure when N_g == N
  SUM   — "career total of X points"  -> missing-value bound Delta+ and a
                                         tolerance checksum on the assembled sum

A claim is TRUSTED only once corroborated by >= 2 independent DOMAINS with
agreeing values (wrong claims are contained, not believed — §5.2 end).
A corroborated COUNT also arms the cardinality hard brake (App. E) and,
when short, emits gap-directed formulations. Certificates granted on
pre-ER counts can be REVOKED by the post-ER second checksum pass
(impl guide §14.1 — counts mean nothing until clusters settle).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config
from .frontier import Formulation
from .types import Claim, sha16


@dataclass
class ChecksumStatus:
    certified: bool = False
    kind: str = ""                  # "COUNT" | "SUM"
    gap: int = 0                    # records still missing per the COUNT claim
    delta_plus: float | None = None # SUM missing-value bound (>= 0)


class ClaimsEngine:
    def __init__(self):
        self._claims: list[Claim] = []

    def ingest(self, claim: Claim) -> None:
        self._claims.append(claim)

    def corroborated(self, stratum: str, kind: str) -> float | None:
        """A claim value backed by >= 2 distinct domains (echo on one domain
        is one vote — corroboration counts evidence, not copies, §4.3)."""
        by_value: dict[float, set] = defaultdict(set)
        for c in self._claims:
            if c.stratum == stratum and c.kind == kind:
                by_value[c.value].add(c.domain)
        supported = sorted(((len(d), v) for v, d in by_value.items()
                            if len(d) >= 2), reverse=True)
        if not supported:
            return None
        # Conflicting values TIED at the top support level: a contradiction,
        # not a corroboration — refuse rather than pick arbitrarily (§5.2 end:
        # wrong claims are contained, not trusted).
        if len(supported) > 1 and supported[0][0] == supported[1][0]:
            return None
        return supported[0][1]

    def checksum(self, stratum: str, n_records: int, assembled_sum: int
                 ) -> ChecksumStatus:
        """Run both checksums against the assembled view of the stratum."""
        st = ChecksumStatus()
        count = self.corroborated(stratum, "COUNT")
        if count is not None:
            if n_records == int(count):
                st.certified, st.kind = True, "COUNT"   # exact closure (Thm 4)
            elif n_records < int(count):
                st.gap = int(count) - n_records          # targeted search fodder
        total = self.corroborated(stratum, "SUM")
        if total is not None:
            st.delta_plus = max(total - assembled_sum, 0.0)  # missing-value bound
            rel = abs(total - assembled_sum) / max(total, 1.0)
            # SUM may certify only when no corroborated COUNT contradicts the
            # record count: a checksum at odds with another checksum is a
            # contradiction to flag, never a certificate (webaggr.pdf §5.2 end)
            count_agrees = count is None or n_records == int(count)
            if not st.certified and rel <= config.TOL_REL and count_agrees:
                st.certified, st.kind = True, "SUM"      # tolerance checksum
        return st

    def gap_formulations(self, stratum: str, player_name: str, gap: int,
                         known_seasons: set[str]) -> list[Formulation]:
        """A residual COUNT gap becomes targeted searches (§5.2: 'a residual
        gap becomes a targeted search'). Seasons we hold tell us which eras
        to probe: search just before the earliest and after the latest."""
        probes = [f"{player_name} rookie season points",
                  f"{player_name} season by season stats early career",
                  f"{player_name} final NBA season stats"]
        if known_seasons:
            first = min(known_seasons)
            probes.append(f"{player_name} points before {first[:4]}")
        fms = []
        for q in probes[: max(gap, 1) + 1]:
            fms.append(Formulation(
                formulation_id=sha16(f"gap|{stratum}|{q}")[:8], query=q,
                p_success=0.5, yield_if_success=float(gap),
                stratum=stratum, gap_directed=True))
        return fms

    def claimed_count(self, stratum: str) -> int | None:
        c = self.corroborated(stratum, "COUNT")
        return None if c is None else int(c)
