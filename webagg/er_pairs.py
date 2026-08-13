"""ER labeled pairs: sampling, storage, and the fitted-matcher loader (A2).

Paper grounding (webaggr_sigmod.pdf Sec. 6): "A calibrated matcher returns
theta(x, y) = Pr[x == y | psi(x, y)] ... Use CalibratedClassifierCV on a
200-pair hand-labeled set; calibration is what makes the per-pair error
alpha a number." The guide adds the labeling recipe: "Mix easy positives
(exact match) and easy negatives (clearly different companies), plus hard
cases from the escalation band. Save the labeled file under
data/ground_truth/match_pairs.csv."

This module supplies the three missing pieces around the already-implemented
Matcher.fit():

  1. SAMPLING (`sample_pairs`): draw a stratified labeling queue from a real
     run's mentions -- easy-positive / easy-negative / band buckets, judged
     by the COLD matcher's theta (before any labels exist there is nothing
     else to stratify by; the band bucket is literally the guide's "hard
     cases from the escalation band"). Pairs come from the SAME blocking
     (`candidate_pairs`) the live pass uses: alpha is the matcher's error on
     candidate pairs, so pairs blocking would never propose are out of scope
     by construction.

  2. STORAGE (data/ground_truth/match_pairs.csv, the guide's path): each row
     freezes the pair's FIVE FEATURE VALUES at sampling time, next to the
     human context (surfaces, passages, domains, kinds) and an empty label
     cell the review CLI fills in.

     Deviation (documented): the guide never says how a fitted matcher
     reaches the pipeline. We freeze features into the CSV and refit from
     them at pipeline start -- the exact pattern the conformal gate already
     uses (load_calibration_set + gate.fit, milliseconds on 200 rows,
     deterministic given the file). Recomputing features at fit time would
     instead require every historical run DB plus torch at every startup.
     Consequence to be aware of: if features() itself changes, stored
     vectors are stale -- bump FEATURE_NAMES and re-sample (the loader
     refuses a CSV whose header disagrees).

  3. LOADING (`load_fitted_matcher`): the pipeline-facing constructor.
     Enough labels (config.ER_MIN_LABELED, both classes >=
     config.ER_MIN_PER_CLASS) -> a FITTED matcher with a real alpha;
     otherwise the cold-start Matcher, loudly, never a crash -- bootstrap
     semantics identical to the unfitted gate.

     Deviation (documented): the fitting floor (40 pairs, >= 10 per class)
     is ours -- the guide targets ~200 pairs but names no minimum, and
     Platt-scaling a 5-pair "set" would produce a meaningless alpha wearing
     a certified costume. Below the floor we prefer the honest cold start.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .entity_resolution import Matcher, candidate_pairs, features

# The five features, IN ORDER, as features() returns them. The CSV header
# embeds these names; a mismatch at load time means the stored vectors were
# frozen by a different feature set and must not silently feed .fit().
FEATURE_NAMES = ["f_name_sim", "f_part_sim", "f_same_domain",
                 "f_emb_cos", "f_temporal"]

# Full column order of match_pairs.csv (context first, features, then label).
COLUMNS = (["pair_id", "mention_a", "mention_b", "surface_a", "surface_b",
            "kind_a", "kind_b", "domain_a", "domain_b",
            "passage_a", "passage_b"]
           + FEATURE_NAMES
           + ["theta_cold", "bucket", "run_db", "label"])


def _pair_id(mid_a: str, mid_b: str) -> str:
    """Stable id for an unordered mention pair (dedupe key across re-runs)."""
    a, b = sorted((mid_a, mid_b))
    return hashlib.sha1(f"{a}|{b}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 1. Sampling the labeling queue
# --------------------------------------------------------------------------- #

def sample_pairs(mentions: list, source_lookup: dict, *,
                 n_easy_pos: int = 60, n_easy_neg: int = 60,
                 n_band: int = 80, seed: int = 0,
                 matcher: Optional[Matcher] = None,
                 run_db: str = "") -> list[dict]:
    """Draw the stratified labeling queue from one run's mentions.

    Buckets by the COLD matcher's theta against its own thresholds:
        theta >= tau_plus   -> easy_pos   (expected same:      confirm cheaply)
        theta <= tau_minus  -> easy_neg   (expected different: confirm cheaply)
        in between          -> band       (the hard cases -- oversample these;
                                           they are where alpha is earned)
    Bucket names record where a pair CAME from, not its label -- the human's
    verdict is the only label, and disagreements with the bucket are exactly
    the informative rows.

    Deterministic under `seed` (numpy RandomState + sorted candidate order),
    so a re-run over the same DB proposes the same queue.
    """
    matcher = matcher or Matcher()      # cold by construction pre-labels
    by_id = {m.mention_id: m for m in mentions}
    rng = np.random.RandomState(seed)

    scored: dict[str, list[dict]] = {"easy_pos": [], "easy_neg": [], "band": []}
    for (a, b) in sorted(candidate_pairs(mentions, source_lookup)):
        m_a, m_b = by_id[a], by_id[b]
        x = features(m_a, m_b, source_lookup)
        theta = matcher.score(x)
        bucket = ("easy_pos" if theta >= matcher.tau_plus else
                  "easy_neg" if theta <= matcher.tau_minus else "band")
        scored[bucket].append({
            "pair_id": _pair_id(a, b),
            "mention_a": a, "mention_b": b,
            "surface_a": m_a.entity_surface, "surface_b": m_b.entity_surface,
            "kind_a": m_a.record_kind, "kind_b": m_b.record_kind,
            "domain_a": source_lookup[m_a.source_id].domain,
            "domain_b": source_lookup[m_b.source_id].domain,
            "passage_a": (m_a.passage or "")[:240],
            "passage_b": (m_b.passage or "")[:240],
            **{name: f"{val:.6f}" for name, val in zip(FEATURE_NAMES, x)},
            "theta_cold": f"{theta:.6f}",
            "bucket": bucket,
            "run_db": run_db,
            "label": "",                # the human fills this in
        })

    want = {"easy_pos": n_easy_pos, "easy_neg": n_easy_neg, "band": n_band}
    out: list[dict] = []
    for bucket, rows in scored.items():
        take = min(want[bucket], len(rows))
        idx = rng.choice(len(rows), size=take, replace=False) if rows else []
        out.extend(rows[i] for i in sorted(idx))
    return out


# --------------------------------------------------------------------------- #
# 2. The CSV on disk
# --------------------------------------------------------------------------- #

def load_rows(path: Path) -> list[dict]:
    """All rows of match_pairs.csv ([] when absent -- bootstrap semantics).

    Refuses a header that disagrees with COLUMNS: stored feature vectors
    from an older feature set must never silently feed .fit().
    """
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"{path} header does not match the current feature set; "
                f"re-sample the queue (expected {COLUMNS}, "
                f"got {reader.fieldnames})")
        return list(reader)


def write_rows(path: Path, rows: list[dict]) -> None:
    """(Re)write the whole CSV -- small file, atomic-enough for one user."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def append_rows(path: Path, new_rows: list[dict]) -> tuple[int, int]:
    """Merge new queue rows into the CSV, deduped on pair_id.

    Existing rows WIN (their label may already be filled in; a re-sample
    must never clobber human work). Returns (n_added, n_skipped) so the
    CLI can print an honest '+N new' (the idempotency-dedup lesson: a
    re-run over the same DB reports +0 instead of doubling the file).
    """
    have = load_rows(path)
    seen = {r["pair_id"] for r in have}
    added = [r for r in new_rows if r["pair_id"] not in seen]
    write_rows(path, have + added)
    return len(added), len(new_rows) - len(added)


def labeled_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Labeled rows -> (X, y) for Matcher.fit(). Unlabeled rows are ignored.

    Labels are '1' (same) / '0' (different) as the review CLI writes them;
    anything else in the cell (a stray 's', whitespace) is treated as
    unlabeled rather than guessed at.
    """
    X, y = [], []
    for r in rows:
        lab = (r.get("label") or "").strip()
        if lab in ("0", "1"):
            X.append([float(r[name]) for name in FEATURE_NAMES])
            y.append(int(lab))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


# --------------------------------------------------------------------------- #
# 3. The pipeline-facing loader
# --------------------------------------------------------------------------- #

def load_fitted_matcher(path: Path | None = None, *,
                        tau_minus: float = 0.15,
                        tau_plus: float = 0.85) -> Matcher:
    """Construct the run's Matcher: FITTED when the labeled set is big enough,
    cold-start otherwise -- never a crash (mirror of the unfitted gate).

    "Big enough" = config.ER_MIN_LABELED labeled pairs overall AND
    config.ER_MIN_PER_CLASS in EACH class: CalibratedClassifierCV needs both
    classes in every fold, and an alpha estimated from a handful of pairs
    would be noise wearing a certified costume (documented deviation: the
    guide names no floor; we refuse to fit below one).
    """
    m = Matcher(tau_minus=tau_minus, tau_plus=tau_plus)
    rows = load_rows(path if path is not None else config.MATCH_PAIRS)
    X, y = labeled_matrix(rows)
    n_pos = int((y == 1).sum()) if len(y) else 0
    n_neg = int((y == 0).sum()) if len(y) else 0
    if (len(y) >= config.ER_MIN_LABELED
            and min(n_pos, n_neg) >= config.ER_MIN_PER_CLASS):
        m.fit(X, y)
        print(f"[er] matcher fitted on {len(y)} labeled pairs "
              f"({n_pos} same / {n_neg} different); alpha={m.alpha:.4f}")
    else:
        print(f"[er] cold-start matcher: {len(y)} labeled pairs "
              f"({n_pos}/{n_neg}) < floor "
              f"({config.ER_MIN_LABELED}, >={config.ER_MIN_PER_CLASS} "
              f"per class); alpha uncertified")
    return m
