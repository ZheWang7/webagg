"""Entity resolution for the NBA domain (webaggr.pdf §6) — deterministic.

Why no fitted matcher here (a PRE-REGISTERED simplification, disclosed in
PREREGISTRATION_EXP2S.md): the finance domain lacked a join key, so ER was a
learned probabilistic matcher with bounded error. Here the record key is
(player, season) and the season IS a join key once canonicalized; the only
ER work left is folding name VARIANTS of the one queried player onto one
canonical entity ('J. Brunson', 'Jalen Brunson Jr.' -> 'jalen brunson').

Cross-player contamination (another player sharing the surname on the same
page) is handled by the guard below, not by merging.
"""
from __future__ import annotations

from . import canon


def resolve_entity(entity_surface: str, target_player: str) -> str | None:
    """Map a mention's surface name onto the run's target player, or None
    if it is a different player (contamination guard, webaggr.pdf §6/§10).

    Merge rule (deterministic, pre-registered):
      full normalized match            -> merge
      last name + first initial match  -> merge ('J. Brunson')
      anything else                    -> reject (never silently merge)
    """
    target = canon.normalize_name(target_player)
    surf = canon.normalize_name(entity_surface)
    if not surf:
        return None
    if surf == target:
        return target
    t_parts, s_parts = target.split(), surf.split()
    if len(t_parts) >= 2 and len(s_parts) >= 2:
        same_last = s_parts[-1] == t_parts[-1]
        initial_ok = s_parts[0][0] == t_parts[0][0]
        if same_last and initial_ok:
            return target                      # 'j brunson' -> 'jalen brunson'
    if len(s_parts) == 1 and s_parts[0] == t_parts[-1]:
        return target                          # bare surname on a profile page
    return None                                # different player: contamination
