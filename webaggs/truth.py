"""Truth table build, freeze and EXACT grading (webaggr.pdf §9 oracle harness).

Differences from the finance oracle, by design (PREREGISTRATION_EXP2S.md):
  * two-sided completeness — every played season is in the official record,
    so there is no reject taxonomy and no one-sided criterion;
  * exact-match grading on (player, season) -> integer points — no amount
    tolerance machinery.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .canon import normalize_name


@dataclass
class GradeReport:
    player: str
    n_truth: int
    n_asserted: int
    n_matched_exact: int        # (season, points) exactly right
    n_season_hit: int           # season found, any value (record recall basis)
    n_spurious: int             # asserted seasons not in truth
    record_recall: float        # season_hit / n_truth
    exact_recall: float         # matched_exact / n_truth
    true_total: int
    asserted_total: int
    rel_agg_error: float        # |asserted - true| / true


def truth_paths(data_dir: Path, cohort: str):
    base = data_dir / "truth" / cohort
    return base / "truth.csv", base / "manifest.json"


def write_truth(data_dir: Path, cohort: str, rows: list[dict]) -> Path:
    """rows: {player, player_id, season, pts, gp}. Writes truth.csv and a
    freeze manifest with its SHA-256 — the frozen answer key (append-only
    after any entity has been run; pre-registration discipline)."""
    csv_path, man_path = truth_paths(data_dir, cohort)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [(normalize_name(r["player"]), r["season"]) for r in rows]
    if len(keys) != len(set(keys)):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise RuntimeError(f"Duplicate (player, season) rows in truth input: "
                           f"{dupes[:5]} — a frozen answer key must have one "
                           "record per key; refusing to write a corrupt key.")
    rows = sorted(rows, key=lambda r: (normalize_name(r["player"]), r["season"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["player", "player_id", "season",
                                          "pts", "gp"])
        w.writeheader()
        w.writerows(rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    players = sorted({normalize_name(r["player"]) for r in rows})
    man_path.write_text(json.dumps(
        {"cohort": cohort, "sha256": digest, "n_rows": len(rows),
         "players": players}, indent=2), encoding="utf-8")
    return csv_path


def load_truth(data_dir: Path, cohort: str, *, verify: bool = True):
    """-> {player_norm: {season: pts}}. Verifies the freeze hash by default."""
    csv_path, man_path = truth_paths(data_dir, cohort)
    if not csv_path.exists():
        raise FileNotFoundError(f"No truth table at {csv_path}; run "
                                "scripts/build_cohort.py first.")
    if verify:
        man = json.loads(man_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if digest != man["sha256"]:
            raise RuntimeError(
                f"Truth table {csv_path} does not match its freeze manifest — "
                "the answer key changed after freezing. Rebuild under a NEW "
                "cohort name; never edit a frozen key.")
    truth: dict[str, dict[str, int]] = {}
    n_rows = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            p = normalize_name(row["player"])
            truth.setdefault(p, {})[season_key(row["season"])] = int(row["pts"])
    if n_rows != sum(len(s) for s in truth.values()):
        raise RuntimeError(f"Truth table {csv_path} has duplicate "
                           "(player, season) rows — a corrupt answer key; "
                           "rebuild it rather than letting last-row-wins "
                           "silently decide the truth.")
    return truth


def load_display_names(data_dir: Path, cohort: str) -> dict[str, str]:
    """-> {player_norm: official display name}. Live runs search with the
    official casing ('Jalen Brunson', not 'jalen brunson') — better queries,
    same stratum key (the pipeline normalizes internally)."""
    csv_path, _ = truth_paths(data_dir, cohort)
    names: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[normalize_name(row["player"])] = row["player"]
    return names


def season_key(season_id: str) -> str:
    """Oracle SEASON_ID is already 'YYYY-YY'; normalize defensively."""
    from .canon import canonical_season
    s = canonical_season(season_id)
    if s is None:
        raise ValueError(f"Unparseable oracle season id: {season_id!r}")
    return s


def grade(player: str, truth_for_player: dict[str, int],
          asserted: dict[str, int]) -> GradeReport:
    """EXACT grading: a record scores only if (season, points) both match.
    record_recall uses season-hit (the discovery side the stopping rule
    certifies); exact_recall folds in value fidelity."""
    seasons_t = set(truth_for_player)
    seasons_a = set(asserted)
    season_hit = seasons_t & seasons_a
    exact = {s for s in season_hit if asserted[s] == truth_for_player[s]}
    true_total = sum(truth_for_player.values())
    asserted_total = sum(asserted.values())
    return GradeReport(
        player=player, n_truth=len(seasons_t), n_asserted=len(seasons_a),
        n_matched_exact=len(exact), n_season_hit=len(season_hit),
        n_spurious=len(seasons_a - seasons_t),
        record_recall=len(season_hit) / max(len(seasons_t), 1),
        exact_recall=len(exact) / max(len(seasons_t), 1),
        true_total=true_total, asserted_total=asserted_total,
        rel_agg_error=abs(asserted_total - true_total) / max(true_total, 1))
