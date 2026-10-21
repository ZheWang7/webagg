"""Split-conformal extraction gate (webaggr.pdf §7.1; impl guide delta_E).

The extractor reports a confidence in [0,1] per mention. The gate accepts a
mention only when its nonconformity score (1 - confidence) is at or below
the (1 - delta_E) empirical quantile fitted on a LABELED calibration set of
(confidence, correct) pairs — so at most ~delta_E of correct extractions are
wrongly rejected, the usual split-conformal direction.

Cold-start refusal: with no fitted calibration file the gate REFUSES to run
(mirrors the finance harness refusing bootstrap modes). A live run without
a calibration set would silently void the delta_E term in the interval.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from . import config


class ConformalError(RuntimeError):
    pass


class ConformalGate:
    def __init__(self, threshold: float, n_cal: int):
        self.threshold = threshold      # max accepted nonconformity (1 - conf)
        self.n_cal = n_cal

    def accepts(self, confidence: float) -> bool:
        return (1.0 - confidence) <= self.threshold

    @staticmethod
    def fit(pairs: list[tuple[float, bool]], delta_e: float = config.DELTA_E
            ) -> "ConformalGate":
        """Fit on labeled (confidence, correct) pairs. Scores from CORRECT
        extractions define the quantile — the conformal convention."""
        scores = sorted(1.0 - c for c, ok in pairs if ok)
        if len(scores) < 20:
            raise ConformalError(
                f"Calibration set too small ({len(scores)} correct examples; "
                "need >= 20). Label more extraction outcomes first — the gate "
                "refuses cold starts by design.")
        n = len(scores)
        # split-conformal rank: ceil((n+1)(1-delta_E)) / n -th order statistic
        k = min(n, math.ceil((n + 1) * (1.0 - delta_e)))
        return ConformalGate(threshold=scores[k - 1], n_cal=n)

    @staticmethod
    def load(path: str | Path | None = None) -> "ConformalGate":
        """Load pairs from CSV (columns: confidence,correct) and fit."""
        p = Path(path) if path else config.DATA_DIR / "conformal" / "calibration.csv"
        if not p.exists():
            raise ConformalError(
                f"No conformal calibration file at {p}. Create it by labeling "
                "extraction outcomes (see README §Calibration) — refusing to "
                "run with an unfitted gate.")
        pairs = []
        with open(p, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f), start=2):
                try:
                    pairs.append((float(row["confidence"]),
                                  row["correct"].strip().lower()
                                  in ("1", "true", "yes")))
                except (KeyError, TypeError, ValueError, AttributeError) as e:
                    raise ConformalError(
                        f"Malformed calibration file {p} at line {i}: needs "
                        f"columns confidence,correct ({e!r}). Fix the file — "
                        "the gate never guesses.") from e
        return ConformalGate.fit(pairs)
