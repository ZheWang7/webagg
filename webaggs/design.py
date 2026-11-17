"""Design arithmetic for the statistical stop (webaggr.pdf §3.2-3.3).

Pure functions, no I/O: they answer "can the certificate fire at all with
these parameters?" BEFORE any money is spent. The stop test needs

    U_hat + psi < eps,   psi = Y_CAP*(1+beta)*sqrt(2*tau_max*ln(1/delta_M)) / N

and, with two paginated indexes walked one page per occasion, the walk
itself costs occasions. Both must fit inside tau_max.

Symbols: N = distinct records found, u = singleton fraction f1/N left
after both walks (records only ONE index has), overhead = occasions spent
on non-index searches (seeds, follow-ups).
"""
from __future__ import annotations

import math


def psi_at(n: int, y_cap: int, tau_max: int, beta: float = 1.0,
           delta_m: float = 0.05) -> float:
    """The conservative radius the live stop test uses, at N = n."""
    return (y_cap * (1 + beta)
            * math.sqrt(2 * tau_max * math.log(1 / delta_m)) / max(n, 1))


def n_needed(eps: float, y_cap: int, tau_max: int, beta: float = 1.0,
             delta_m: float = 0.05, u: float = 0.0) -> float:
    """Smallest N at which u + psi < eps can hold (inf if u >= eps:
    the singleton fraction alone already exceeds the target)."""
    room = eps - u                      # what is left for psi
    if room <= 0:
        return math.inf
    return psi_at(1, y_cap, tau_max, beta, delta_m) / room


def pages_for(n: float, y_cap: int, u: float = 0.0) -> float:
    """Index pages (= occasions) to walk N distinct records through two
    indexes: doubly covered records cost 2 page slots, singletons 1."""
    return (2 - u) * n / y_cap


def feasible_band(eps: float, y_cap: int, tau_max: int, beta: float = 1.0,
                  delta_m: float = 0.05, u: float = 0.0,
                  overhead: int = 20):
    """-> (n_min, n_max) of N for which the certificate CAN fire, or None.
    n_min: psi small enough. n_max: both walks still fit inside tau_max
    (the frontier stays hot until every page is issued, so an unfinished
    walk blocks the stop)."""
    lo = n_needed(eps, y_cap, tau_max, beta, delta_m, u)
    hi = y_cap * (tau_max - overhead) / (2 - u)
    return (lo, hi) if lo < hi else None


def min_tau(eps: float, beta: float = 1.0, delta_m: float = 0.05,
            u: float = 0.0, overhead: int = 20, slack: float = 1.25) -> int:
    """Smallest tau_max whose feasible band is at least `slack` wide
    (n_max >= slack * n_min). Independent of the page size."""
    for tau in range(50, 20001, 10):
        band = feasible_band(eps, 1, tau, beta, delta_m, u, overhead)
        if band and band[1] >= slack * band[0]:
            return tau
    raise ValueError("no tau_max <= 20000 gives a feasible band")
