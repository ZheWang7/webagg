"""Harvest the conformal-gate calibration set from the truth table (Sec. 15).

Closes seam A1: the gate's machinery (calibration.py, guide Sec. 6.3) has
been real since ch. 6, but with no file at config.CALIBRATION_SET it runs
in bootstrap accept-all mode and delta_E is vacuous. Sec. 6.3's instruction
is "maintain a labeled calibration set of (passage, true value) pairs from
the DEPLOYMENT DOMAIN" -- so the division of labor here is:

    text source : a REAL open-web agent run (the deployment distribution --
                  news pages, blogs, press releases about a calibration
                  entity), run with --deny sec like every experiment run;
    labels      : two-tier. The Sec. 15 truth table auto-decides what it
                  can -- canonical equality with any totalAmountSold a
                  filing ever carried (PRE-amendment values count), or
                  within CLAIM_TOL_REL of one (the checksum layer's own
                  agreement threshold, so a faithful read of a rounded
                  press figure labels correct). Mentions FARTHER than the
                  tolerance go to a human review queue: the open web
                  faithfully reports IPO proceeds, cumulative totals and
                  full-round figures that Form D does not contain, and the
                  registry cannot tell those apart from hallucinations.
                  This is Sec. 6.3's "label ~100 mentions once" shrunk to
                  the tail the registry can't settle (roughly half, on the
                  first live harvest).

Why pre-amendment values count as correct: the gate certifies READING
fidelity (did the extractor read the page right), not world-truth -- a
faithful read of a page reporting the original round size is a correct
extraction; supersession (Sec. 8) is the layer whose job it is to kill that
value later. Labeling it an error here would punish the reader for the
page.

Residual label noise, and its direction: a page that itself misreports a
number is read faithfully but labels WRONG here (no registry value matches).
That inflates the error count -> pushes the conformal threshold DOWN ->
the gate abstains more. Conservative for delta_E, costly only in
abstentions; the receipt reports the within-2% share (checksum tolerance,
config.CLAIM_TOL_REL) so rounding-style "errors" are visible at a glance.

Three integrity guards, each refusing a way the calibration could lie:

  1. CIRCULARITY: every harvested mention must carry the gate_uncalibrated
     stamp. Harvesting from a run that already had a fitted gate would see
     only accepted (high-confidence) mentions -- a truncated distribution
     calibrating the very gate that truncated it.
  2. SPLIT LEAKAGE: only entities in the cohort manifest's CALIBRATION
     half may be harvested. Validation entities grade the experiments; a
     gate tuned on them poisons every downstream claim.
  3. CONTAMINATION: the harvest run must have been a DENIED run (its DB
     must self-describe a denylist covering sec.gov) and its artifact must
     still pass assert_no_denied_sources. Registry pages are trivially
     easy to extract from; letting them into the calibration set would
     break exchangeability with the denied experiment runs the gate will
     actually guard.

Scope note (documented approximation): only `amount` mentions are
harvested -- the registry gives authoritative USD values, but no clean
counterpart for announcement dates or stage names. Deployment applies the
one threshold to all attributes; per-attribute gates are a possible later
refinement, not required by the guide.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

from . import config
from .calibration import ConformalGate, load_calibration_set, nonconf
from .canonicalize import canonicalize_value
from .denylist import Denylist, assert_no_denied_sources
from .frontier import normalize_surface
from .storage import MeasurementRow, MentionRow

# The guide's "label ~100+ mentions once" (Sec. 6.3). Below this the file is
# still written (a small honest set beats bootstrap accept-all) but the
# receipt and manifest carry a loud warning.
MIN_RECOMMENDED_N = 100


# --------------------------------------------------------------------------- #
# scope: which mentions belong to this entity
# --------------------------------------------------------------------------- #

def surface_in_scope(surface: str, aliases: list[str]) -> bool:
    """True iff the mention's surface names the harvested entity.

    Normalized containment either way: 'SpaceX' matches 'SpaceX rocket
    division' and 'Space Exploration Technologies Corp' matches its own
    substrings. Loose on purpose -- ER's job is disambiguation; here we
    only need to keep OTHER companies' rounds out of the label pool.
    """
    ns = normalize_surface(surface or "")
    if not ns:
        return False
    for a in aliases:
        na = normalize_surface(a or "")
        if na and (na in ns or ns in na):
            return True
    return False


# --------------------------------------------------------------------------- #
# labels: the legitimate value set from the oracle DB
# --------------------------------------------------------------------------- #

def legit_amounts(truth_session, entity_name: str) -> list[tuple[str, float]]:
    """Every amount ANY filing of this entity carried, from the oracle's
    provenance mentions (extractor_id='oracle', so each value is an XML tag
    lookup, never a guess). Returns (canonical value string, numeric) pairs.
    """
    rows = (truth_session.query(MentionRow)
            .filter(MentionRow.attribute == "amount",
                    MentionRow.extractor_id == "oracle",
                    MentionRow.entity_surface == entity_name).all())
    out, seen = [], set()
    for r in rows:
        if r.value_num is not None and r.value_num not in seen:
            seen.add(r.value_num)
            out.append((r.value, float(r.value_num)))
    return out


def _num(mention_value: str, value_num) -> float | None:
    """The mention's numeric reading: trust the canonical value_num stamped
    at extraction; fall back to the pipeline-wide canonicalizer."""
    if value_num is not None:
        return float(value_num)
    try:
        return float(canonicalize_value(mention_value))
    except (ValueError, TypeError):
        return None


def label_mention(value: str, value_num,
                  legit: list[tuple[str, float]]) -> tuple[str, str, float]:
    """One mention -> (paired true value, label_source, rel_distance).

    Three-way outcome. The first LIVE harvest (35 real press mentions:
    1 exact, 18 within tolerance, 16 far) showed that exact equality is
    the wrong rule for open-web text, so:

      registry_exact -- canonical equality with a filed value;
      registry_tol   -- within CLAIM_TOL_REL of a filed value. This is the
                        checksum layer's OWN definition of agreement, so
                        the gate's labels can't be stricter than the
                        pipeline they protect: a faithful read of "$1.2B"
                        for a filed 1,188,241,352 is a correct reading.
                        true := pred, so nonconf() scores it correct;
      review         -- farther than the tolerance. The registry CANNOT
                        decide these: the open web faithfully reports IPO
                        proceeds, cumulative raised-to-date totals, full
                        round sizes where the filing covers one tranche,
                        and convertibles -- none of which Form D contains
                        -- and a hallucination looks exactly the same from
                        here. Routed to the human review queue with the
                        nearest filed value riding along as context.

    Unparseable values go to review too (a garbled extraction is precisely
    the thing a human should glance at, and dropping it would bias the set
    toward easy examples).
    """
    cv = canonicalize_value(value or "")
    for tv, _ in legit:
        if cv == canonicalize_value(tv):
            return tv, "registry_exact", 0.0
    pn = _num(value, value_num)
    if pn is None:
        return legit[0][0], "review", float("inf")
    tv, tn = min(legit, key=lambda x: abs(pn - x[1]))
    rel = abs(pn - tn) / max(abs(tn), 1e-9)
    if rel <= config.CLAIM_TOL_REL:
        # correct BY THE SYSTEM'S OWN TOLERANCE: assert the prediction as
        # the true reading so the gate's equality test agrees with ours
        return value, "registry_tol", rel
    return tv, "review", rel


# --------------------------------------------------------------------------- #
# the three integrity guards
# --------------------------------------------------------------------------- #

def check_split(manifest: dict, entity_id: str) -> None:
    """Guard 2: calibration half only, and the entity must exist at all."""
    split = manifest.get("split", {})
    if entity_id in split.get("validation", []):
        raise RuntimeError(
            f"{entity_id} is in the cohort's VALIDATION half -- it grades "
            "the experiments and must never feed calibration (guide Sec. "
            "15: split into calibration and validation halves).")
    if entity_id not in split.get("calibration", []):
        raise RuntimeError(
            f"{entity_id} is not in the cohort manifest's calibration "
            f"half {split.get('calibration')} -- rebuild the cohort or "
            "check the entity id.")


def check_run_conditions(run_session, *, allow_undenied: bool = False) -> None:
    """Guard 3: the harvest run mirrored experiment conditions.

    The run DB must SELF-DESCRIBE a denylist covering sec.gov (the
    denylist_active measurement written by run_query), and the artifact
    must still scan clean -- the same two-sided proof the grading harness
    uses. allow_undenied=True downgrades a missing denylist to a warning
    for exploratory sets; the caller must stamp that into the manifest.
    """
    rows = (run_session.query(MeasurementRow)
            .filter(MeasurementRow.metric == "denylist_active").all())
    suffixes = set()
    for r in rows:
        suffixes.update((r.extra or {}).get("suffixes", []))
    if "sec.gov" not in suffixes:
        msg = ("harvest run was NOT a denied run (no denylist_active row "
               "covering sec.gov): its extraction distribution may include "
               "registry pages and does not match denied experiment runs.")
        if not allow_undenied:
            raise RuntimeError(msg + " Re-run with --deny sec, or pass "
                                     "--allow-undenied to accept the "
                                     "mismatch explicitly.")
        warnings.warn(msg)
    # re-verify on the artifact regardless -- enforcement is not trusted,
    # it is re-checked (same doctrine as the grading harness)
    assert_no_denied_sources(run_session, Denylist(["sec"]))


def gather_mentions(run_session, aliases: list[str]) -> tuple[list, dict]:
    """Guard 1 + scoping: accepted `amount` mentions of the entity, every
    one of which must carry the gate_uncalibrated stamp."""
    rows = (run_session.query(MentionRow)
            .filter(MentionRow.attribute == "amount",
                    MentionRow.accepted == True).all())      # noqa: E712
    stats = {"amount_mentions": len(rows), "out_of_scope": 0}
    kept = []
    for r in rows:
        if not surface_in_scope(r.entity_surface, aliases):
            stats["out_of_scope"] += 1
            continue
        if "gate_uncalibrated" not in (r.validator_flags or []):
            raise RuntimeError(
                f"mention {r.mention_id} was accepted by a FITTED gate -- "
                "harvesting from a gated run would calibrate the gate on "
                "the distribution the gate itself truncated. Use a "
                "bootstrap-mode run (no calibration file present).")
        kept.append(r)
    return kept, stats


# --------------------------------------------------------------------------- #
# harvest + persist
# --------------------------------------------------------------------------- #

def harvest(run_session, truth_session, *, run_id: str, entity_id: str,
            entity_name: str, aliases: list[str]) -> tuple[list[dict],
                                                           list[dict], dict]:
    """The full labeling pass for one (run, entity).

    Returns (auto_rows, review_rows, stats):
      auto_rows   -- registry-decided (exact or within tolerance), ready
                     for the calibration file. Row shape is a SUPERSET of
                     what load_calibration_set reads; the extra keys are
                     provenance the loader ignores.
      review_rows -- registry-UNDECIDABLE, destined for the human queue.
                     They carry the passage, the source URL and the
                     nearest filed value so each judgment is a ten-second
                     read; 'true' is deliberately absent until a human
                     decides it (guide Sec. 6.3's "label ~100 mentions
                     once", shrunk to the tail the registry can't settle).
    """
    legit = legit_amounts(truth_session, entity_name)
    if not legit:
        raise RuntimeError(
            f"no oracle amounts for entity_name={entity_name!r} in the "
            "truth DB -- check the name (it must equal the filed "
            "entityName) or rebuild the cohort.")
    mentions, stats = gather_mentions(run_session, aliases)
    # source_id -> url, so a reviewer can open the page behind a passage
    from .storage import SourceRow
    urls = {s.source_id: s.url for s in run_session.query(SourceRow).all()}
    auto, review = [], []
    for m in mentions:
        true, label_source, rel = label_mention(m.value, m.value_num, legit)
        row = {"pred": m.value, "self_conf": float(m.self_conf),
               "mention_id": m.mention_id, "run_id": run_id,
               "entity_id": entity_id, "label_source": label_source,
               "rel_dist": None if rel == float("inf") else rel}
        if label_source == "review":
            row.update({"registry_nearest": true,
                        "passage": m.passage or "",
                        "url": urls.get(m.source_id, "")})
            review.append(row)
        else:
            row["true"] = true
            auto.append(row)
    stats.update({"harvested": len(auto) + len(review),
                  "auto_exact": sum(r["label_source"] == "registry_exact"
                                    for r in auto),
                  "auto_tol": sum(r["label_source"] == "registry_tol"
                                  for r in auto),
                  "queued_for_review": len(review)})
    return auto, review, stats


def append_calibration(path: str | Path, new_rows: list[dict]) -> tuple[int, int]:
    """Append rows to the calibration JSON, idempotent by mention_id --
    re-running the harvester on the same run adds nothing twice. Returns
    (n_added, n_total)."""
    p = Path(path)
    existing = json.loads(p.read_text()) if p.exists() else []
    seen = {r.get("mention_id") for r in existing if r.get("mention_id")}
    added = [r for r in new_rows if r["mention_id"] not in seen]
    merged = existing + added
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2))
    return len(added), len(merged)


def append_queue(queue_path: str | Path, cal_path: str | Path,
                 rows: list[dict]) -> tuple[int, int]:
    """Append registry-undecidable rows to the review queue, idempotent by
    mention_id against BOTH the queue and the calibration file (a mention
    already decided by a human must never reappear for re-review)."""
    qp, cp = Path(queue_path), Path(cal_path)
    queue = json.loads(qp.read_text()) if qp.exists() else []
    decided = set()
    if cp.exists():
        decided = {r.get("mention_id") for r in json.loads(cp.read_text())}
    seen = decided | {r.get("mention_id") for r in queue}
    added = [r for r in rows if r["mention_id"] not in seen]
    queue += added
    qp.parent.mkdir(parents=True, exist_ok=True)
    qp.write_text(json.dumps(queue, indent=2))
    return len(added), len(queue)


def record_decision(cal_path: str | Path, queue_path: str | Path,
                    mention_id: str, faithful: bool) -> dict:
    """Apply one human judgment and move the row queue -> calibration file.

    faithful=True  -> the passage really says this: true := pred, the gate
                      learns it as a CORRECT extraction (label_source
                      human_faithful);
    faithful=False -> the extractor garbled the page: true := the nearest
                      filed value, so nonconf()'s distance term reflects
                      how far off it was (label_source human_error).

    Saves both files immediately (crash-safe: quitting mid-review loses
    nothing already answered). Returns the calibration row written.
    """
    qp = Path(queue_path)
    queue = json.loads(qp.read_text()) if qp.exists() else []
    idx = next((i for i, r in enumerate(queue)
                if r["mention_id"] == mention_id), None)
    if idx is None:
        raise KeyError(f"mention {mention_id} not in the review queue")
    row = queue.pop(idx)
    cal_row = {"pred": row["pred"],
               "true": row["pred"] if faithful else row["registry_nearest"],
               "self_conf": row["self_conf"],
               "mention_id": row["mention_id"], "run_id": row["run_id"],
               "entity_id": row["entity_id"], "rel_dist": row["rel_dist"],
               "label_source": "human_faithful" if faithful
                               else "human_error"}
    append_calibration(cal_path, [cal_row])
    qp.write_text(json.dumps(queue, indent=2))
    return cal_row


def threshold_preview(path: str | Path, delta_E: float = None) -> dict:
    """Fit a throwaway gate on the file exactly as run_query will and report
    what the threshold MEANS: deployment scores are 1 - self_conf <= 1, so
    t_hat >= 1 is calibrated-accept-all, while t_hat < 1 rejects every
    mention with self_conf < 1 - t_hat."""
    cal = load_calibration_set(path)
    gate = ConformalGate(delta_E=delta_E or config.DELTA_E).fit(cal)
    t = gate.threshold()
    return {"n": len(cal), "threshold": t,
            "min_self_conf_accepted": max(0.0, 1.0 - t),
            "accepts_everything": t >= 1.0,
            "scores_ge_1": int(sum(nonconf(p, tr, c) >= 1.0
                                   for p, tr, c in cal))}
