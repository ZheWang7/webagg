"""Seam A2 (ER labeled pairs) -- sampling, storage, loader. Offline.

One test per duty (repo convention):
  test_sample_buckets_respect_thresholds -> stratification follows cold theta
  test_sample_deterministic_under_seed   -> same DB + seed => same queue
  test_append_dedupes_and_keeps_labels   -> re-sampling never clobbers work
  test_header_mismatch_refuses           -> stale feature vectors can't fit
  test_labeled_matrix_ignores_unlabeled  -> only 0/1 cells become y
  test_loader_below_floor_cold_start     -> too few labels => honest alpha=None
  test_loader_fits_above_floor           -> floor met => fitted, alpha real
  test_fitted_alpha_reaches_erresult     -> alpha flows into resolve_entities

No torch, no network: the embedder is stubbed exactly as in test_er_ch9.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import webagg.entity_resolution as er                      # noqa: E402
from webagg import config                                  # noqa: E402
from webagg.er_pairs import (COLUMNS, append_rows,         # noqa: E402
                             labeled_matrix, load_fitted_matcher, load_rows,
                             sample_pairs, write_rows)
from webagg.type_defs import Mention, Source               # noqa: E402

_NOW = datetime(2025, 1, 1)


class _FakeEncoder:
    def encode(self, name: str) -> np.ndarray:
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(16)


@pytest.fixture(autouse=True)
def _stub_embedder():
    er._embed = _FakeEncoder()          # embedder() returns this, never torch
    yield
    er._embed = None


def _src(sid: str, domain: str = "example.com") -> Source:
    return Source(source_id=sid, url=f"https://{domain}/a", domain=domain,
                  fetch_time=_NOW, publish_time=None, title=None,
                  main_text="test", formulation_id="f0")


def _men(mid: str, sid: str, surface: str) -> Mention:
    return Mention(mention_id=mid, source_id=sid, entity_surface=surface,
                   record_kind="funding_round", attribute="amount",
                   value="$1M", passage=f"passage about {surface}",
                   extracted_at=_NOW)


def _world():
    """A tiny blocked world: near-duplicate Acmes (high theta), an unrelated
    same-prefix company (low/band theta), across two domains."""
    sources = {"s1": _src("s1", "a.com"), "s2": _src("s2", "a.com"),
               "s3": _src("s3", "b.com")}
    mentions = [_men("m1", "s1", "Acme Corp"),
                _men("m2", "s2", "Acme Corp"),        # exact dup -> easy_pos
                _men("m3", "s3", "Acme, Inc."),
                _men("m4", "s3", "Acme Logistics")]   # shares the block only
    return mentions, sources


# --------------------------------------------------------------------------- #
def test_sample_buckets_respect_thresholds():
    mentions, sources = _world()
    rows = sample_pairs(mentions, sources, n_easy_pos=10, n_easy_neg=10,
                        n_band=10, seed=0)
    assert rows, "blocking proposed no candidate pairs"
    m = er.Matcher()                    # the same cold thresholds
    for r in rows:
        th = float(r["theta_cold"])
        want = ("easy_pos" if th >= m.tau_plus else
                "easy_neg" if th <= m.tau_minus else "band")
        assert r["bucket"] == want
        assert r["label"] == ""         # sampling never pre-labels


def test_sample_deterministic_under_seed():
    mentions, sources = _world()
    a = sample_pairs(mentions, sources, seed=3)
    b = sample_pairs(mentions, sources, seed=3)
    assert a == b                       # same DB + same seed => same queue


def test_append_dedupes_and_keeps_labels(tmp_path: Path):
    mentions, sources = _world()
    csv_path = tmp_path / "match_pairs.csv"
    rows = sample_pairs(mentions, sources, seed=0)
    added, skipped = append_rows(csv_path, rows)
    assert (added, skipped) == (len(rows), 0)

    # a human labels the first row...
    stored = load_rows(csv_path)
    stored[0]["label"] = "1"
    write_rows(csv_path, stored)

    # ...then the queue is re-sampled over the same DB: +0 new, label intact
    added2, skipped2 = append_rows(csv_path, sample_pairs(mentions, sources,
                                                          seed=0))
    assert added2 == 0 and skipped2 == len(rows)
    assert load_rows(csv_path)[0]["label"] == "1"


def test_header_mismatch_refuses(tmp_path: Path):
    p = tmp_path / "match_pairs.csv"
    p.write_text("pair_id,old_feature,label\nx,0.5,1\n")
    with pytest.raises(ValueError):     # stale feature set must not fit
        load_rows(p)


def test_labeled_matrix_ignores_unlabeled():
    row = {name: "0.5" for name in COLUMNS}
    r1 = dict(row, label="1")
    r2 = dict(row, label="0")
    r3 = dict(row, label="")            # unlabeled
    r4 = dict(row, label="s")           # stray key-press: NOT a label
    X, y = labeled_matrix([r1, r2, r3, r4])
    assert X.shape == (2, 5) and list(y) == [1, 0]


# --------------------------------------------------------------------------- #
def _csv_with_labels(tmp_path: Path, n_pos: int, n_neg: int) -> Path:
    """A synthetic labeled CSV whose classes are linearly separable in one
    feature -- enough for CalibratedClassifierCV to fit cleanly."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        feats = {"f_name_sim": f"{(0.9 if pos else 0.1) + rng.normal(0, .03):.6f}",
                 "f_part_sim": "0.5", "f_same_domain": "0.0",
                 "f_emb_cos": "0.5", "f_temporal": "1.0"}
        rows.append({**{c: "" for c in COLUMNS}, **feats,
                     "pair_id": f"p{i}", "label": "1" if pos else "0"})
    p = tmp_path / "match_pairs.csv"
    write_rows(p, rows)
    return p


def test_loader_below_floor_cold_start(tmp_path: Path):
    p = _csv_with_labels(tmp_path, n_pos=5, n_neg=5)   # 10 < ER_MIN_LABELED
    m = load_fitted_matcher(p)
    assert m.clf is None and m.alpha is None           # honest cold start
    m2 = load_fitted_matcher(tmp_path / "absent.csv")  # missing file: same
    assert m2.clf is None and m2.alpha is None


def test_loader_fits_above_floor(tmp_path: Path):
    n = max(config.ER_MIN_LABELED // 2, config.ER_MIN_PER_CLASS) + 5
    p = _csv_with_labels(tmp_path, n_pos=n, n_neg=n)
    m = load_fitted_matcher(p)
    assert m.clf is not None                           # fitted
    assert m.alpha is not None and 0.0 <= m.alpha < 0.5
    # and the fitted classifier actually separates the two feature regimes
    hi = m.score(np.array([0.9, 0.5, 0.0, 0.5, 1.0]))
    lo = m.score(np.array([0.1, 0.5, 0.0, 0.5, 1.0]))
    assert hi > 0.5 > lo


def test_fitted_alpha_reaches_erresult(tmp_path: Path):
    n = max(config.ER_MIN_LABELED // 2, config.ER_MIN_PER_CLASS) + 5
    m = load_fitted_matcher(_csv_with_labels(tmp_path, n_pos=n, n_neg=n))
    mentions, sources = _world()
    res = er.resolve_entities(mentions, m, sources,
                              adjudicator=lambda a, b, look: 0.9)
    assert res.alpha == m.alpha         # Sec.-13's input rides the ERResult
