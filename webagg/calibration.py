"""The conformal gate (impl guide §6.3): stage 4, the statistical gatekeeper.

Split-conformal, distribution-free (paper Prop. 2). The idea in one line:
score every mention by "how unlike a known-correct extraction does this
look" (nonconformity); calibrate a threshold on labeled examples; accept
only mentions under the threshold. The guarantee needs only that the
calibration set and deployment mentions are EXCHANGEABLE -- no model of
the LLM's errors -- and says: among accepted mentions, at most a delta_E
fraction are wrong, marginally. Abstentions are an observable COST, never
an error.

Recalibrate per domain and per extractor version: distribution shift
breaks the exchangeability the guarantee needs.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

from .canonicalize import canonicalize_value
from .type_defs import Mention


def nonconf(pred: str, true: str, self_conf: float) -> float:
    """Nonconformity score for a LABELED calibration pair (guide §6.3).

    Correct prediction -> 1 - self_conf (a confident correct answer
    conforms best). Wrong prediction -> >= 1, growing with the relative
    numeric distance when both sides are numbers (a wrong-by-2x value is
    less conforming than wrong-by-1%), capped at 2.0 so one wild outlier
    cannot dominate the sort.
    """
    p, t = canonicalize_value(pred), canonicalize_value(true)
    if p == t:
        return 1.0 - float(self_conf)
    try:
        pn, tn = float(p), float(t)
        rel = abs(pn - tn) / max(abs(tn), 1e-9)
        return min(1.0 + rel, 2.0)
    except ValueError:
        return 2.0                                  # non-numeric and wrong


def nonconf_pred(m: Mention) -> float:
    """Deployment-time score, where truth is unknown: 1 - self_conf.

    The singleton condition of Prop. 2 ({v : s(x,v) <= t_hat} has exactly
    one member) is enforced UPSTREAM by dual extraction: only A/B-agreed
    values reach the gate, so the candidate set is already a singleton.
    """
    return 1.0 - float(m.self_conf)


class ConformalGate:
    def __init__(self, delta_E: float = 0.05):
        self.delta_E = delta_E
        self.scores: np.ndarray | None = None   # sorted calibration scores

    def fit(self, cal: list[tuple[str, str, float]]) -> "ConformalGate":
        """cal: list of (predicted_value, true_value, self_conf) from the
        deployment domain (label ~100+ mentions once; store as JSON)."""
        self.scores = np.sort([nonconf(p, t, c) for (p, t, c) in cal])
        return self

    def threshold(self) -> float:
        """t_hat = the ceil((1 - delta_E)(n+1))-th smallest calibration
        score (paper Prop. 2). The +1 is the finite-sample correction that
        makes the guarantee exact rather than asymptotic."""
        n = len(self.scores)
        k = int(np.ceil((1 - self.delta_E) * (n + 1))) - 1   # 0-indexed
        return float(self.scores[min(k, n - 1)])

    @property
    def fitted(self) -> bool:
        return self.scores is not None and len(self.scores) > 0

    def accept(self, m: Mention) -> bool:
        """Accept iff the nonconformity of the (already singleton) value is
        within the calibrated threshold; otherwise the extractor abstains.

        BOOTSTRAP BEHAVIOR (deliberate, documented): an UNFITTED gate
        accepts everything and stamps 'gate_uncalibrated' into
        validator_flags. This keeps the pipeline runnable before you have
        labeled a calibration set, while making every such mention
        auditable -- no silent fake guarantee. Once a calibration file
        exists, the flag disappears and delta_E means what it says.
        """
        if not self.fitted:
            if "gate_uncalibrated" not in m.validator_flags:
                m.validator_flags = list(m.validator_flags) + ["gate_uncalibrated"]
            return True
        return nonconf_pred(m) <= self.threshold()


def load_calibration_set(path: str | Path) -> list[tuple[str, str, float]]:
    """Read a labeled calibration set from JSON:
        [{"pred": "...", "true": "...", "self_conf": 0.9}, ...]
    Returns [] when the file does not exist (-> bootstrap gate)."""
    p = Path(path)
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    return [(r["pred"], r["true"], float(r["self_conf"])) for r in rows]
