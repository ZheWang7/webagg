"""Per-record corroboration: count evidence, not copies (webaggr.pdf §4).

SCOPE NOTE (pre-registered): in this domain every web source ultimately
mirrors ONE origin (the official scorer), so 'independent confirmation' in
the Theorem 2 sense does not exist — Exp 2-S makes NO corroboration claims.
What survives here is the echo-collapse discipline: identical values on the
same DOMAIN are one vote, and votes are weighted by source-class reliability
(config.CLASS_PRIORS), so a scraped blog cannot outvote basketball-reference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from . import config


@dataclass
class ResolvedRecord:
    record_key: str          # "<player_norm>|<season>"
    season: str
    value: int               # adopted points value
    belief: float            # winner weight minus runner-up weight (margin)
    n_domains: int           # distinct domains behind the adopted value


def corroborate_record(record_key: str, season: str, mentions, sources
                       ) -> ResolvedRecord:
    """Adopt one value per (player, season) record.

    Votes: one per (value, domain) pair — repeats on a domain collapse to one
    (echo invariance in its cheapest form, §4.3) — weighted by class prior.
    """
    weight: dict[int, float] = defaultdict(float)
    domains: dict[int, set] = defaultdict(set)
    for m in mentions:
        src = sources[m.source_id]
        if src.domain in domains[m.value]:
            continue                             # echo on one domain: one vote
        domains[m.value].add(src.domain)
        weight[m.value] += config.CLASS_PRIORS.get(src.source_class, 0.4)

    ranked = sorted(weight.items(), key=lambda kv: kv[1], reverse=True)
    best_value, best_w = ranked[0]
    runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
    return ResolvedRecord(record_key=record_key, season=season,
                          value=best_value, belief=best_w - runner_w,
                          n_domains=len(domains[best_value]))
