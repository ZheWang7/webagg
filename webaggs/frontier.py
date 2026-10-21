"""Frontier state and the stopping statistics (webaggr.pdf §3.2-3.3).

Faithful to the implementation guide's frontier.py:
  * covered[record_key] = set of FORMULATION ids — capture OCCASIONS, never
    sources. (Pitfall #1: counting sources collapses f1 and the agent stops
    early with a clean certificate.)
  * U_hat = f1/N + beta * (sum of residual yields of unissued formulations) / N
    — frontier credit is a SUM OF YIELDS, not a count (eq. 1).
  * psi = sqrt(2 * V_max * log(1/(delta_M * w_g))) with per-occasion increment
    bound c = Y_CAP*(1+beta)/n (Theorem 1 / Theorem 2 radius).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from . import config


@dataclass
class Formulation:
    formulation_id: str
    query: str
    p_success: float                 # LLM estimate: P(>=1 new record)
    yield_if_success: float          # LLM estimate: E[new records | success]
    cost: float = config.SEARCH_COST
    stratum: str | None = None
    gap_directed: bool = False       # came from the claims engine (§5.2)
    issued: bool = False
    realized_yield: int = 0

    @property
    def residual_yield(self) -> float:
        """\\hat y(q) in records; 0 once issued (webaggr.pdf Def. 5)."""
        return 0.0 if self.issued else self.p_success * self.yield_if_success

    def reservation_index(self, lam: float) -> float:
        """Economic ordering (webaggr.pdf App. B): heuristic benefit-per-cost.
        Ordering only — never a certificate."""
        if self.issued or self.p_success <= 0:
            return -math.inf
        return lam * self.yield_if_success - self.cost / self.p_success


@dataclass
class StratumState:
    name: str
    certified: str = ""              # "" | "checksum" — "" not None: pandas
    #                                  coerces None->NaN and NaN is truthy
    #                                  (the exp3 replay bug), so empty string.
    claimed_count: int | None = None # corroborated COUNT claim (App. E brake)


@dataclass
class FrontierState:
    formulations: dict[str, Formulation] = field(default_factory=dict)
    # covered[record_key] = set of FORMULATION ids that surfaced it (occasions!)
    covered: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    record_stratum: dict[str, str] = field(default_factory=dict)
    strata: dict[str, StratumState] = field(default_factory=dict)
    T: int = 0                       # capture occasions issued so far
    Y_CAP: int = config.Y_CAP
    BETA: float = config.BETA

    # --- per-stratum pool statistics -----------------------------------
    def _records(self, pool: set[str]):
        return [occ for rk, occ in self.covered.items()
                if self.record_stratum.get(rk) in pool]

    def N(self, pool) -> int:
        return len(self._records(pool))

    def f(self, k: int, pool) -> int:
        """f_k: records covered by exactly k occasions (f1 = singletons)."""
        return sum(1 for occ in self._records(pool) if len(occ) == k)

    def U_hat(self, pool) -> float:
        """Good-Turing-Frontier estimator (webaggr.pdf §3.2 eq. 1)."""
        n = self.N(pool)
        if n == 0:
            return 1.0                          # nothing found: all mass unseen
        credit = sum(fm.residual_yield for fm in self.formulations.values()
                     if not fm.issued and (fm.stratum in pool or fm.stratum is None))
        return self.f(1, pool) / n + self.BETA * credit / n

    def psi(self, pool, delta_m: float, w_g: float, max_occasions: int) -> float:
        """Anytime-valid radius (Thm 1 via Ville; Thm 2 splits delta by w_g)."""
        n = max(self.N(pool), 1)
        c = self.Y_CAP * (1 + self.BETA) / n    # per-occasion increment bound c_t
        v_max = max_occasions * c * c           # conservative V bound
        return math.sqrt(2 * v_max * math.log(1 / (delta_m * w_g)))

    # --- the stop test ---------------------------------------------------
    def uncertified_strata(self) -> list[str]:
        return [g for g, s in self.strata.items() if not s.certified]

    def stratum_pools(self):
        for g in self.strata:
            yield g, {g}                        # Exp 2-S: one stratum per run

    def hot_frontier(self, g: str, eta: float) -> bool:
        """Conjunct (ii): an unissued formulation still promises yield >= eta."""
        return any(not f.issued and f.residual_yield >= eta
                   and f.stratum in (g, None)
                   for f in self.formulations.values())


def all_strata_pass(state: FrontierState, eps_g: float, delta_m: float,
                    eta: float, max_steps: int) -> bool:
    """The stop test (webaggr.pdf §3.3 Def. 6 + Thm 2), exactly as the guide:
    per uncertified stratum, (U_hat + psi < eps_g) AND frontier not hot,
    with the corroborated-cardinality hard brake (App. E) checked first.
    NOTE eps_g is read ONLY here — that is what makes offline replay at a
    looser epsilon valid from the same recorded ledger (Exp 2 design)."""
    n_strata = max(len(state.strata), 1)
    w_g = 1.0 / n_strata                        # equal deviation-budget split
    for g, pool in state.stratum_pools():
        s = state.strata[g]
        if s.certified:
            continue                            # checksum-closed strata exit P
        if s.claimed_count is not None and state.N(pool) < s.claimed_count:
            return False                        # cardinality hard brake
        u = state.U_hat(pool)
        psi = state.psi(pool, delta_m, w_g, max_steps)
        if not (u + psi < eps_g) or state.hot_frontier(g, eta):
            return False                        # the TWO conjuncts (§3.3)
    return True
