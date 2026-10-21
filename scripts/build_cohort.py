"""Build and FREEZE the Exp 2-S truth table from the official NBA oracle.

Usage (PowerShell, from the repo root):
    python scripts/build_cohort.py --cohort nba_v1 --candidates cohort/candidates.txt

candidates.txt: one player name per line; '#' comments allowed. Ambiguous
names (two players share it) make the oracle RAISE — disambiguate in the
file before rerunning; the script never guesses.

Output:
    data/truth/<cohort>/truth.csv + manifest.json (frozen: sha256)
    data/truth/<cohort>/cohort_report.csv  — career games per player, so the
    pre-registered selection rule (stars vs <=N-game fringe) is applied to
    OFFICIAL numbers, before any live run.

Pre-registration discipline: freeze BEFORE the first live run; after that
the cohort is append-only under a NEW cohort name.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                       # noqa: E402
from webaggs.oracle_nba import NBAStatsDriver    # noqa: E402
from webaggs.truth import parse_candidate_line, write_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--candidates", required=True)
    args = ap.parse_args()

    names = [ln.strip() for ln in Path(args.candidates).read_text(
        encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")]
    if not names:
        raise SystemExit("Candidates file is empty.")

    driver = NBAStatsDriver()
    rows, report = [], []
    for line in names:
        name, pid = parse_candidate_line(line)
        if pid is None:                       # no explicit id: oracle lookup
            pid, official = driver.find_player_id(name)
        else:                                 # 'Name | ID' bypasses the flaky
            official = name                   # lookup; name taken as written
        seasons = driver.career_regular_season(pid)
        games = sum(r["gp"] for r in seasons)
        print(f"  {official}: {len(seasons)} seasons, {games} games, "
              f"{sum(r['pts'] for r in seasons)} pts")
        report.append({"player": official, "player_id": pid,
                       "n_seasons": len(seasons), "career_games": games,
                       "career_pts": sum(r["pts"] for r in seasons)})
        for r in seasons:
            rows.append({"player": official, "player_id": pid,
                         "season": r["season"], "pts": r["pts"], "gp": r["gp"]})

    csv_path = write_truth(config.DATA_DIR, args.cohort, rows)
    rep_path = csv_path.parent / "cohort_report.csv"
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"\nFrozen truth table: {csv_path}")
    print(f"Cohort report (apply the pre-registered selection rule): {rep_path}")


if __name__ == "__main__":
    main()
