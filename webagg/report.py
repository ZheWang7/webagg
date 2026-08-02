"""The report layer: the TWO-TERM honest interval (impl guide §14.2,
design paper Theorem 6 / Corollary 1).

The paper's final interval, per stratum g and globally, is

    SUM_hat_g  +/-  (eps_C^g + eps_F) * SUM_hat_g

exactly two terms and a regime label -- NOT the old four-part bracket
(reading and join error now live INSIDE eps_F, measured together by the
§13 fidelity certificate):

  * eps_C^g -- the COMPLETENESS slack of group g, regime-dependent:
        registry   -> 0                       (Theorem 3: delta_F = 0 over K*)
        COUNT      -> 1 - belief              (Theorem 4b, conditional on the claim)
        SUM        -> Delta+_g/SUM + (1-b)    (Theorem 4a value-gap + claim belief)
        statistical-> eps_g                   (Theorem 1, read as VALUE-recall under
                                               a stated bound on any one record's
                                               share of the group total)
  * eps_F   -- the domain-wide FIDELITY level, identical for every group
               (§13's Learn-Then-Test certificate; App.-H-style fallback
               constant, clearly labelled, when no certificate exists).

A group that earned no certificate is not given a number it did not earn:
it reports ABANDONED with its ACHIEVED U_hat_g + psi_g (guide: "if your
CLI prints one number with one interval, you have not implemented the
paper").

REPO DEVIATION (documented): the guide sketches this inline in
end_to_end(); it lives in its own module so it is pure compute --
importable and testable offline with hand-built states, and reusable by
the CLI (scripts/run_query.py) without touching the pipeline.
"""
from __future__ import annotations

import math
from collections import defaultdict

from . import config
from .frontier import FrontierState, w_g


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _usd(cv) -> float | None:
    """Numeric dollars of an adopted value: the canonical value_num if the
    corroborator stored one, else a best-effort float parse, else None
    (non-numeric cells never crash the aggregate -- they just don't sum)."""
    if cv is None:
        return None
    if getattr(cv, "value_num", None) is not None:
        return float(cv.value_num)
    try:
        return float(str(cv.value).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return None


def resolve_eps_F(*, eps_F=None, domain: str | None = None):
    """Pick the fidelity level and REMEMBER where it came from.

    Priority: explicit override > stored §13 certificate for the domain >
    config.EPS_F_FALLBACK. The provenance tag ends up in the printed
    label, so a fallback can never silently impersonate a calibrated
    certificate (the §14 decision posed in risk_control.load_fidelity_cert).
    """
    if eps_F is not None:
        return float(eps_F), "given"
    if domain:
        from .risk_control import load_fidelity_cert   # lazy: pulls in scipy
        cert = load_fidelity_cert(domain)
        if cert is not None:
            return float(cert), "ltt"                  # calibrated (§13)
    return float(config.EPS_F_FALLBACK), "fallback"    # App.-H-style constant


def _statistical_status(state: FrontierState, g: str, *, eps_g: float,
                        delta_M: float, max_steps: int):
    """Re-evaluate conjunct (i) of the stopping rule for one stratum AT
    REPORT TIME: U_hat_g + psi_g < eps_g, plus the App. E cardinality
    brake. Legitimate post hoc because psi is ANYTIME-valid -- the radius
    holds at every step, including the last one.

    Returns (passed: bool, achieved: float U_hat+psi, n: int).
    """
    S = state.strata[g]
    pool = {g}
    n = state.N(pool)
    U = state.U_hat(pool)
    psi = state.psi(pool, delta_M, w_g(state, g), max_steps,
                    V_realized=S.V or None)
    achieved = U + psi
    passed = achieved < eps_g
    # App. E hard brake: a corroborated COUNT claim of more records than we
    # found forbids a statistical pass no matter what U_hat says.
    if S.claimed_count is not None and n < S.claimed_count:
        passed = False
    return passed, achieved, n


# ---------------------------------------------------------------------------
# the aggregate (guide §14.2 pseudocode, adapted to the repo's shapes)
# ---------------------------------------------------------------------------

def aggregate_two_term(resolved: list[dict], state: FrontierState | None, *,
                       mode: str = "open_web",
                       aggregate_attr: str = "amount",
                       eps_g: float = config.EPS_G,
                       delta_M: float = config.DELTA_M,
                       max_steps: int = config.MAX_STEPS,
                       eps_F: float | None = None,
                       domain: str | None = None) -> dict:
    """Per-group totals with the two-term interval (Cor. 1).

    resolved: the pipeline's record dicts ({"entity_id", "record_kind",
        "attributes": {attr: CorroboratedValue}, ...}).
    state:    the run's FrontierState (post-ER re-key), carrying each
        stratum's certification fields. None in schema mode (the registry
        certificate covers everything) and on legacy fixture paths.

    Returns {group: row} plus "__global__" (value-weighted combine over
    certified groups) and "__abandoned__" (uncertified groups with their
    achieved U_hat+psi -- honesty, not a completeness number).
    """
    eps_F_val, eps_F_src = resolve_eps_F(eps_F=eps_F, domain=domain)

    # ---- group the resolved records by stratum (= entity_id post-ER) ------
    recs_by_g: dict[str, list[dict]] = defaultdict(list)
    for r in resolved:
        recs_by_g[r["entity_id"]].append(r)
    # strata the state knows about but that yielded no resolved record still
    # get a row (a certified-empty group is a statement, not a bug)
    groups = set(recs_by_g)
    if state is not None:
        groups |= set(state.strata)

    out: dict[str, dict] = {}
    abandoned: dict[str, dict] = {}

    for g in sorted(groups):
        recs = recs_by_g.get(g, [])
        vals = [v for v in (_usd(r["attributes"].get(aggregate_attr))
                            for r in recs) if v is not None]
        total = float(sum(vals))

        # ---- pick the certifying regime for eps_C^g ----------------------
        S = state.strata.get(g) if state is not None else None
        if mode == "schema":
            # Theorem 3: the sweep enumerated the key universe; delta_F = 0
            # over the addressable closure -- zero completeness slack.
            eps_C, label, kind, belief = 0.0, "registry (delta_F=0)", "REGISTRY", 1.0
        elif S is not None and S.certified == "registry":
            eps_C, label, kind, belief = 0.0, "registry (delta_F=0)", "REGISTRY", 1.0
        elif S is not None and S.certified == "checksum" and S.cert_kind == "COUNT":
            # Theorem 4(b): every record is in hand; residual risk is only
            # that the certifying claim itself was wrong -> (1 - belief).
            b = S.cert_belief or 0.0
            eps_C, kind, belief = (1.0 - b), "COUNT", b
            label = f"checksum COUNT (b={b:.2f})"
        elif S is not None and S.certified == "checksum" and S.cert_kind == "SUM":
            # Theorem 4(a): value gap Delta+_g (relative) + claim belief.
            b = S.cert_belief or 0.0
            dp = S.cert_delta_plus or 0.0
            eps_C, kind, belief = dp / max(total, 1e-9) + (1.0 - b), "SUM", b
            label = f"checksum SUM (D+={dp:,.0f}, b={b:.2f})"
        elif S is not None:
            # no constraint certificate: did the STATISTICS earn eps_g?
            passed, achieved, n_g = _statistical_status(
                state, g, eps_g=eps_g, delta_M=delta_M, max_steps=max_steps)
            if passed:
                eps_C, kind, belief = eps_g, "STATISTICAL", None
                label = f"statistical (eps={eps_g})"
            else:
                # ABANDONED: no completeness number -- report what was
                # actually achieved (guide §14.4's honesty requirement).
                abandoned[g] = {"total": total, "n": len(recs),
                                "U_plus_psi": achieved, "n_seen": n_g}
                continue
        else:
            # legacy/fixture path: open-web resolved records with no state.
            # Nothing to certify against -> statistical slack, labelled so.
            eps_C, kind, belief = eps_g, "STATISTICAL", None
            label = f"statistical (eps={eps_g}, no state)"

        halfwidth = (eps_C + eps_F_val) * total    # the WHOLE interval: two terms
        kappas = [(_k if _k is not None else 0)
                  for _k in (r["attributes"][aggregate_attr].kappa
                             for r in recs
                             if aggregate_attr in r["attributes"])]
        out[g] = {"total": total, "n": len(recs),
                  "eps_C": eps_C, "eps_F": eps_F_val,
                  "halfwidth": halfwidth, "certificate": label,
                  # machine-readable regime fields (verify.py reads these):
                  "regime": kind, "cert_belief": belief,
                  # forgery margin kappa: per-cell DIAGNOSTIC (Prop. 1),
                  # printed next to the answer, never inside the interval
                  "min_kappa": min(kappas) if kappas else None}

    # ---- global combine: value-weighted eps_C, one shared eps_F ----------
    tot = sum(row["total"] for row in out.values())
    if tot > 0:
        eps_C_glob = sum(row["eps_C"] * row["total"] for row in out.values()) / tot
    else:
        eps_C_glob = 0.0
    result: dict = dict(out)
    result["__global__"] = {
        "total": tot, "n": sum(row["n"] for row in out.values()),
        "eps_C": eps_C_glob, "eps_F": eps_F_val, "eps_F_source": eps_F_src,
        "halfwidth": sum(row["halfwidth"] for row in out.values()),
    }
    result["__abandoned__"] = abandoned
    return result


# ---------------------------------------------------------------------------
# the CLI table (guide §14.4) -- a per-group TABLE, never one number
# ---------------------------------------------------------------------------

def format_report(report: dict, verify_menu: list[dict] | None = None,
                  stop_reason: str | None = None) -> str:
    """Render the §14.4 table as a string (the CLI prints it; tests read it).

    One line per group with its OWN interval and regime label; ABANDONED
    groups print their achieved U_hat+psi instead of a made-up interval;
    then the global line, the two eps terms with the eps_F provenance tag,
    and the top human checks from the verification allocator.
    """
    lines = [f"{'group':<18}{'total':>14}{'+/-':>12} certificate"]
    for g, row in report.items():
        if g.startswith("__"):
            continue
        lines.append(f"{g:<18}{row['total']:>14,.0f}{row['halfwidth']:>12,.0f} "
                     f"{row['certificate']} (min kappa={row['min_kappa']})")
    reason = f" ({stop_reason})" if stop_reason else ""
    for g, row in report.get("__abandoned__", {}).items():
        lines.append(f"{g:<18}{row['total']:>14,.0f}{'--':>12} "
                     f"ABANDONED{reason}: achieved U+psi="
                     f"{row['U_plus_psi']:.3f} over n={row['n_seen']}")
    gl = report["__global__"]
    lines.append("")
    if gl["n"] == 0 and report.get("__abandoned__"):
        # nothing certified: "GLOBAL 0 +/- 0" would read as "the answer is
        # zero" -- say what it actually means (first live run's lesson)
        lines.append("GLOBAL: no certified strata (all ABANDONED)")
    else:
        lines.append(f"GLOBAL {gl['total']:,.0f} +/- {gl['halfwidth']:,.0f}")
    lines.append(f"  completeness eps_C={gl['eps_C']:.3f}  "
                 f"fidelity eps_F={gl['eps_F']:.3f} [{gl['eps_F_source']}]")
    if verify_menu:
        lines.append("")
        lines.append("Top human checks:")
        for c in verify_menu:
            drop = "inf" if math.isinf(c["drop"]) else f"{c['drop']:,.0f}"
            lines.append(f"  [{c['kind']:>12}] {c['what']} (-{drop})")
    return "\n".join(lines)
