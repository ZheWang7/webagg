"""
Section 13 -- The Fidelity Certificate (SIGMOD implementation guide, ch. 13;
design paper Sec. 7, Theorems 5-6; App. H for the no-cohort fallback).

This is the module that changed when the paper moved to a TWO-TERM guarantee.
The old plan bounded each stage separately -- relevance (rho_phi), extraction
(delta_E), blocking (delta_B), matching (alpha), corroboration, supersession --
and summed the bounds. The SIGMOD version instead certifies their COMPOSITION,
the map Pi_lambda from fetched sources to contributed records, once,
end-to-end, by distribution-free risk control (Learn-Then-Test):

  1. per-entity loss  L_g in [0,1]   -- how wrong the CONTRIBUTED records'
     aggregate is against ground truth (missing records are the completeness
     term and are excluded here);
  2. Learn-Then-Test  -- treat each candidate configuration lambda as the
     hypothesis "E[L(lambda)] <= eps_F", test it on a calibration cohort with
     a Hoeffding p-value, and keep configurations under a fixed-sequence
     family-wise-error control. The selected lambda then carries
     P(E[L] <= eps_F) >= 1 - delta_F for a future exchangeable entity
     (paper Theorem 5);
  3. the certificate  -- ONE number eps_F per domain, stored on disk, read by
     the report-time interval (guide Sec. 14.2: halfwidth = (eps_C + eps_F) *
     total). Recalibrate only when the domain or a model version changes.

TWO PRACTICAL RULES (guide Sec. 13.3), enforced in code where possible:
  (1) It is a CERTIFICATE, not a tuning run. The fixed-sequence procedure
      spends the cohort's statistical power once: iterate configs in a
      pre-committed order, STOP at the first failure, and never re-pick
      lambda after seeing the result (pitfall 8: re-tuning voids 1-delta_F).
  (2) AUDIT ON HELD-OUT DATA. Keep a validation cohort the calibration never
      touched and report realized L_g on it (evaluation experiment 8). The
      split_cohort()/holdout_report() helpers exist for exactly this.

THE FALLBACK WHEN YOU HAVE NO COHORT (guide Sec. 13.4; paper App. H): if a
domain has no ground-truth cohort you cannot calibrate end-to-end. Then
fall back to the old per-channel numbers summed:
    eps_F_fallback = rho_phi + delta_E + delta_B * v_split/SUM
                     + alpha * v_merge/SUM + delta_C + delta_T.
Looser but assumption-light -- and the ONLY place those per-stage bounds
still appear. Prefer the calibrated certificate whenever a cohort exists.

Seams: run_pipeline(g, lam) and truth(g) are INJECTED callables (same
pattern as relevance_fn/extract_fn in schema_addressable.py). The real
implementations -- the withheld-registry harness -- arrive in Sec. 15;
until then the module is fully testable offline with fakes that pin the
contract.

No online sanity test accompanies this chapter ON PURPOSE (same reasoning
as ch. 12): there is no LLM/search seam here that could gracefully degrade
-- the module is arithmetic plus file I/O. The live end-to-end exercise of
the injected run_pipeline seam is Sec. 15's evaluation experiment 8.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from webagg import config
from webagg.canonicalize import canonicalize_value


# ===========================================================================
# 13.0  Ground truth containers + the money helper
# ===========================================================================
# The guide's pseudocode assumes a `truth_g` object with `.true_sum` and
# records carrying `.amount`. We pin that contract here so build_truth.py
# (Sec. 15) has a fixed shape to produce and the offline tests have a fixed
# shape to fake.

@dataclass(frozen=True)
class TruthRecord:
    """One ground-truth record of an entity (e.g. one Form-D filing).

    `key` is the REGISTRY key used to align a reassembled open-web record
    with its true counterpart -- e.g. an EDGAR accession number, or a
    canonical (record_kind|date) string when the registry key is not
    recoverable from the open web. match_to_truth() matches on this.
    """
    key: str
    amount: float                 # true USD amount of this record
    date: Optional[str] = None    # ISO date, informational only


@dataclass(frozen=True)
class TruthEntity:
    """The full answer key for one calibration entity g (one stratum)."""
    entity_id: str
    records: tuple[TruthRecord, ...]

    @property
    def true_sum(self) -> float:
        """SUM_g over the ground truth -- the loss denominator."""
        return float(sum(r.amount for r in self.records))

    @property
    def true_count(self) -> int:
        return len(self.records)


def usd(cv) -> float:
    """Numeric USD value of a CorroboratedValue (0.0 if unreadable).

    Prefers the canonical numeric form stamped at corroboration time
    (value_num, Sec. 4.1); falls back to parsing the raw adopted string
    through the same canonicalizer the pipeline uses everywhere else, so
    "$40M" and "40000000" agree here exactly as they do in the claims
    engine. A record whose amount cannot be read contributes 0 -- and if it
    is MATCHED to a truth record, that shortfall correctly surfaces as
    fidelity error in fidelity_loss().
    """
    if cv is None:
        return 0.0
    if getattr(cv, "value_num", None) is not None:
        return float(cv.value_num)
    try:
        return float(canonicalize_value(getattr(cv, "value", "") or ""))
    except (TypeError, ValueError):
        return 0.0


# ===========================================================================
# 13.2  The per-entity loss L_g
# ===========================================================================

def default_truth_key(record) -> Optional[str]:
    """Extract the truth-alignment key from a ResolvedRecord.

    Priority: an explicit "registry_key" attribute if the pipeline
    recovered one (e.g. the accession number quoted by a news article);
    otherwise the record's date attribute -- the same (kind|date) key the
    oracle writes into TruthRecord.key when no registry id exists. Returns
    None when neither is present: the record can then never match truth and
    is scored SPURIOUS at full value, which is the conservative direction.
    """
    attrs = getattr(record, "attributes", {}) or {}
    rk = attrs.get("registry_key")
    if rk is not None and getattr(rk, "value", None):
        return str(rk.value)
    dt = attrs.get("date")
    if dt is not None and getattr(dt, "value", None):
        return f"{getattr(record, 'record_kind', '?')}|{dt.value}"
    return None


def match_to_truth(resolved_g, truth_g: TruthEntity,
                   key_fn: Callable = default_truth_key
                   ) -> list[tuple[object, Optional[TruthRecord]]]:
    """Align each resolved record with its true counterpart, or None.

    ONE-TO-ONE by key: every truth record can absorb at most ONE resolved
    record. This is what makes over-splits hurt -- if ER split one true
    round into two resolved records, only one of them pairs with truth and
    the duplicate is scored spurious at full value (an over-merge that
    pulls in a foreign round fails to find a key and is spurious too).
    Truth records that nothing matched are MISSING records: completeness,
    not fidelity, so they simply do not appear in the output (guide 13.2).
    """
    by_key = {t.key: t for t in truth_g.records}
    unused = set(by_key)                     # each truth record usable once
    aligned: list[tuple[object, Optional[TruthRecord]]] = []
    for r in resolved_g:
        k = key_fn(r)
        if k is not None and k in unused:
            unused.discard(k)
            aligned.append((r, by_key[k]))
        else:
            aligned.append((r, None))        # spurious: no true counterpart
    return aligned


def fidelity_loss(resolved_g, truth_g: TruthEntity,
                  key_fn: Callable = default_truth_key) -> float:
    """L_g in [0,1]: relative error of the aggregate over CONTRIBUTED
    records only. Missing records are completeness, not fidelity, so they
    are excluded (guide Sec. 13.2; paper Theorem 5).

    This one number absorbs EVERY downstream channel: a misread amount, a
    mis-join that pulls in a foreign round, a wrong corroborated value, an
    undetected stale echo -- all surface as a larger L_g. That is the
    point: we measure the realized composition, including the cancellations
    a per-stage union bound throws away.
    """
    aligned = match_to_truth(resolved_g, truth_g, key_fn)
    # value the pipeline assembled for records that DO have a true counterpart
    assembled = sum(usd(r.attributes.get("amount"))
                    for (r, t) in aligned if t is not None)
    # what those same true counterparts actually sum to
    true_match = sum(t.amount for (r, t) in aligned if t is not None)
    # records with NO true counterpart (over-merge / hallucination):
    # their full value is error
    spurious = sum(usd(r.attributes.get("amount"))
                   for (r, t) in aligned if t is None)
    err = abs(assembled - true_match) + spurious
    return min(1.0, err / max(truth_g.true_sum, 1e-9))


# ===========================================================================
# 13.3  Learn-Then-Test: certifying eps_F
# ===========================================================================

def hoeffding_p(losses, eps_F: float) -> float:
    """p-value for H0: E[L] > eps_F (small p => we may certify E[L] <= eps_F).

    Losses live in [0,1] by construction, so Hoeffding's inequality applies
    with NO variance or distribution assumption: if the true mean exceeded
    eps_F, seeing a sample mean this far below it has probability at most
    exp(-2 n (eps_F - mean)^2). A sample mean at or above eps_F is no
    evidence at all -> p = 1.
    """
    n, mean = len(losses), float(np.mean(losses))
    return float(np.exp(-2 * n * max(0.0, eps_F - mean) ** 2)) if mean < eps_F else 1.0


def learn_then_test(cohort, configs, eps_F: float, delta_F: float, *,
                    run_pipeline: Callable, truth: Callable,
                    trace: Optional[list] = None):
    """Fixed-sequence test over a PRE-COMMITTED, cheapest-first config list.

    For each candidate lambda in order: run the full pipeline on every
    calibration entity, score the losses, and test "E[L(lambda)] <= eps_F"
    at level delta_F. Certify the prefix that passes; STOP at the first
    failure. Fixed-sequence testing controls the family-wise error, so
    EVERY config in the passing prefix simultaneously carries the 1-delta_F
    guarantee -- the returned lambda (the deepest passing one, per the
    guide's pseudocode) does, and so would the cheaper ones before it.

    PRACTICAL RULE (1) lives here: the stop-at-first-failure is what lets
    the selected lambda keep its guarantee. Do NOT re-order or re-run the
    grid after seeing losses -- that turns the certificate into an ordinary
    tuning run and voids 1 - delta_F (guide pitfall 8).

    run_pipeline(g, lam) -> resolved records of entity g under config lam
    truth(g)             -> TruthEntity answer key for g
    trace (optional)     -> a list you pass in; one dict per tested config
                            is appended (for the write-up / experiment 8).

    Returns (lambda_star, realized mean loss) or None if even the first
    config fails.
    """
    certified = None
    for lam in configs:
        losses = [fidelity_loss(run_pipeline(g, lam), truth(g)) for g in cohort]
        p = hoeffding_p(losses, eps_F)
        if trace is not None:
            trace.append({"lam": lam, "mean_loss": float(np.mean(losses)),
                          "p": p, "passed": bool(p <= delta_F)})
        if p <= delta_F:
            certified = (lam, float(np.mean(losses)))
        else:
            break                    # fixed-sequence: stop on first fail
    return certified


# ---------------------------------------------------------------------------
# Practical rule (2): audit on held-out data (evaluation experiment 8)
# ---------------------------------------------------------------------------

def split_cohort(cohort, seed: int = 0):
    """Deterministically split a cohort into (calibration, validation)
    halves. The validation half must NEVER be seen by learn_then_test --
    realized L_g on it is the honest check of eps_F (guide Sec. 13.3,
    rule 2; Sec. 15 experiment 8).
    """
    idx = np.random.RandomState(seed).permutation(len(cohort))
    half = len(cohort) // 2
    cal = [cohort[i] for i in idx[:half]]
    val = [cohort[i] for i in idx[half:]]
    return cal, val


def holdout_report(losses, eps_F: float) -> dict:
    """Summarize realized losses on the UNTOUCHED validation half.

    mean_loss    -- realized E[L]; the certificate claims this <= eps_F
    rate_within  -- fraction of entities with L_g <= eps_F (the per-entity
                    view experiment 8 reports)
    p_value      -- Hoeffding p re-computed on the holdout: small means the
                    holdout independently supports E[L] <= eps_F
    """
    losses = list(losses)
    return {
        "n": len(losses),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "rate_within": (float(np.mean([l <= eps_F for l in losses]))
                        if losses else float("nan")),
        "p_value": hoeffding_p(losses, eps_F) if losses else 1.0,
    }


# ===========================================================================
# 13.4  The fallback when you have no cohort (paper App. H)
# ===========================================================================

def fallback_eps_F(*, rho_phi: float, delta_E: float, delta_B: float,
                   v_split: float, v_merge: float, alpha: float,
                   delta_C: float, delta_T: float, total_sum: float) -> float:
    """Per-stage analytic bound: the old per-channel numbers, summed.

        eps_F = rho_phi                      (audited relevance false-negatives)
              + delta_E                      (conformal extraction miscoverage)
              + delta_B * v_split / SUM      (blocking misses -> value split off)
              + alpha   * v_merge / SUM      (matcher errors -> value merged in)
              + delta_C                      (corroboration adopts a wrong value)
              + delta_T                      (supersession misses a stale echo)

    v_split / v_merge are the value-at-stake totals of the split/merge
    channels (paper Theorem 7: the bounded-error entity join). This is
    LOOSER than the calibrated certificate -- it sums worst cases and
    ignores cancellations -- but needs no ground truth. It is also the only
    place the per-stage bounds still appear; use it only when no cohort
    exists (guide Sec. 13.4).
    """
    S = max(total_sum, 1e-9)
    return (rho_phi + delta_E
            + delta_B * (v_split / S)
            + alpha * (v_merge / S)
            + delta_C + delta_T)


# ===========================================================================
# 13.4  Storing and re-using the certificate
# ===========================================================================
# eps_F is a DOMAIN CONSTANT: calibrate once per domain (and per
# extractor/model version), store it, and the report-time interval simply
# reads it (guide Sec. 14.2 calls load_fidelity_cert(state.domain)).
# Recalibrate when the domain or a model changes -- exactly as you would
# recalibrate the conformal gate, which is now one knob inside lambda.

@dataclass
class FidelityCertificate:
    """Everything the report layer (and the write-up) needs about eps_F."""
    domain: str                     # e.g. "startup_funding"
    eps_F: float                    # the certified level
    delta_F: float                  # confidence it was certified at
    method: str                     # "ltt" (calibrated) | "fallback" (App. H)
    lam: dict = field(default_factory=dict)   # the selected configuration
    mean_loss: float = float("nan")           # realized mean L on calibration
    n_cal: int = 0                            # calibration cohort size
    # model stamps: a cert calibrated under one extractor does NOT transfer
    # to another (guide 13.4) -- load warns when these drift from config
    model_cheap: str = config.MODEL_CHEAP
    model_strong: str = config.MODEL_STRONG
    created: str = ""               # ISO timestamp, UTC-naive (repo convention)


def _cert_path(domain: str):
    return config.FIDELITY_CERT_DIR / f"{domain}.json"


def save_fidelity_cert(cert: FidelityCertificate) -> str:
    """Persist the certificate as pretty JSON (one file per domain).
    Returns the path written, for logging."""
    if not cert.created:
        cert.created = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    path = _cert_path(cert.domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cert), indent=2))
    return str(path)


def load_fidelity_cert_record(domain: str) -> Optional[FidelityCertificate]:
    """Full stored certificate for a domain, or None if never calibrated.
    Warns (but still returns the cert) if the current model configuration
    differs from the one it was calibrated under -- the honest move is to
    recalibrate, but a stale cert with a warning beats silently inventing
    a number."""
    path = _cert_path(domain)
    if not path.exists():
        return None
    cert = FidelityCertificate(**json.loads(path.read_text()))
    if (cert.model_cheap, cert.model_strong) != (config.MODEL_CHEAP,
                                                 config.MODEL_STRONG):
        warnings.warn(
            f"fidelity certificate for '{domain}' was calibrated under models "
            f"({cert.model_cheap}, {cert.model_strong}) but the pipeline now "
            f"runs ({config.MODEL_CHEAP}, {config.MODEL_STRONG}); "
            f"recalibrate (guide Sec. 13.4).", stacklevel=2)
    return cert


def load_fidelity_cert(domain: str) -> Optional[float]:
    """The number the report layer reads: certified eps_F for the domain,
    or None if no certificate exists (the caller then falls back to the
    App. H bound or refuses to print an interval -- Sec. 14's decision)."""
    cert = load_fidelity_cert_record(domain)
    return cert.eps_F if cert else None
