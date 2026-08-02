"""Spending a human wisely: the verification allocator (impl guide §14.3).

The pipeline ends with a small budget B of HUMAN checks. Each candidate
check is scored by how much interval width it would remove if it came
back clean ("drop"), then plain greedy takes the top B -- with B <= 10
and a few hundred candidates that is all the optimality the paper needs
(coverage-style objectives are submodular; greedy is within (1 - 1/e),
paper Prop. 4's machinery).

Candidate kinds, in the guide's order:
  conflict     -- a claim contradicting confident statistics (rule R3) or a
                  count that matched only via a borderline ER decision.
                  Drop = infinity: conflicts outrank everything, ALWAYS first.
  claim        -- verify the certifying claim of a checksum-closed group;
                  clears (1 - belief) * group_total of interval width.
  merge        -- re-adjudicate a fragile ER pair (decision sat inside the
                  matcher band); clears alpha * value_at_stake.
  value        -- re-read a low-belief or weak-entity-link adopted value;
                  clears (1 - belief) * usd(value).
  supersession -- confirm the newest-version choice where stale echoes of a
                  corrected figure were excluded; weight DELTA_T_VERIFY.
"""
from __future__ import annotations

import math

from . import config
from .report import _usd


def verification_menu(resolved: list[dict], ce, state, *,
                      report: dict | None = None,
                      fragile_pairs=(), alpha: float | None = 0.05,
                      budget: int = config.VERIFY_BUDGET) -> list[dict]:
    """Enumerate candidate human checks, score by interval width removed,
    return the greedy top-`budget` (guide §14.3 pseudocode).

    resolved:      pipeline record dicts (attributes -> CorroboratedValue).
    ce:            the run's ClaimsEngine (post-ER re-key), or None on
                   fixture paths -- conflict items then simply don't exist.
    state:         FrontierState with per-stratum cert fields, or None.
    report:        aggregate_two_term() output; supplies group totals so a
                   claim check knows how much width it clears.
    fragile_pairs: ER decisions inside the matcher band, (a, b, theta).
    alpha:         the matcher's calibrated per-pair error (ERResult.alpha).
                   None on COLD START (matcher never fit on labeled pairs,
                   the tracked uncertified-alpha seam): menu scoring then
                   uses a conservative default -- fine for RANKING checks,
                   but it certifies nothing (only Sec. 13 calibration does).
    """
    c: list[dict] = []
    report = report or {}
    alpha_eff = alpha if alpha is not None else 0.05   # ranking default only

    # ---- conflicts FIRST: drop = inf (rule R3: they outrank both sides) --
    queued_pairs = set()
    if ce is not None:
        for k in ce.conflicts:
            c.append({"kind": "conflict", "what": str(k), "drop": float("inf")})
        for item in ce.verification_queue:      # count-sensitivity fragile pairs
            g, a, b, theta = item
            queued_pairs.add((a, b))
            c.append({"kind": "conflict",
                      "what": f"count of '{g}' hinges on ER pair "
                              f"({a}, {b}) at theta={theta:.2f}",
                      "drop": float("inf")})

    # ---- claim checks: (1 - belief) * group total -----------------------
    if state is not None:
        for g, S in state.strata.items():
            if S.certified == "checksum" and S.cert_belief is not None:
                total = report.get(g, {}).get("total", 0.0)
                c.append({"kind": "claim",
                          "what": f"verify certifying {S.cert_kind} claim "
                                  f"for '{g}'",
                          "drop": (1.0 - S.cert_belief) * total})

    # ---- fragile ER merges: alpha * value at stake ----------------------
    # Value at stake per pair defaults to the mean record value in the
    # aggregate (a wrong merge/split moves roughly one record's worth).
    # When NOTHING certified, the report's global is empty -- but fragile
    # pairs still put real value at risk (the first live run priced them
    # all at -0), so fall back to the mean over ALL resolved numeric cells.
    gl = report.get("__global__", {})
    if gl.get("n"):
        v_bar = gl.get("total", 0.0) / gl["n"]
    else:
        vals = [v for r in resolved
                for v in (_usd(cv) for cv in r["attributes"].values())
                if v is not None]
        v_bar = sum(vals) / len(vals) if vals else 0.0
    for pair in fragile_pairs:
        a, b = pair[0], pair[1]
        if (a, b) in queued_pairs:
            continue                 # already escalated to a conflict above
        c.append({"kind": "merge",
                  "what": f"re-adjudicate ER pair ({a}, {b})",
                  "drop": alpha_eff * v_bar})

    # ---- per-cell checks: low belief / weak link / supersession ---------
    for r in resolved:
        rid = f"{r['entity_id']}/{r['record_kind']}"
        for attr, cv in r["attributes"].items():
            v = _usd(cv) or 0.0
            if (cv.belief < config.VERIFY_BELIEF_FLOOR
                    or "weak_entity_link" in cv.validator_flags):
                c.append({"kind": "value",
                          "what": f"re-read {rid}.{attr} "
                                  f"(belief={cv.belief:.2f})",
                          "drop": (1.0 - cv.belief) * v})
            if cv.n_dead_excluded > 0:
                # a corrected figure: confirm the adopted version really is
                # the newest along the authority chain (design §4.3)
                c.append({"kind": "supersession",
                          "what": f"confirm version choice for {rid}.{attr} "
                                  f"({cv.n_dead_excluded} stale echoes "
                                  f"excluded)",
                          "drop": config.DELTA_T_VERIFY * v})

    # ---- greedy: sort by width removed, take the top B ------------------
    return sorted(c, key=lambda x: -x["drop"])[:budget]
