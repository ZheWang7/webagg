"""Grade finished Exp 2-S runs against the frozen truth table (exact grading).

Offline; reads resolved records from the ledgers. Output:
data/experiments/exp2s/results/grades.csv — one row per player with record
recall, exact recall, spurious count and relative aggregation error, plus
the certificate path (statistical / checksum / abandoned) for the headline
split.

Usage: python scripts/grade_exp2s.py --cohort nba_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                   # noqa: E402
from webaggs.storage import Ledger           # noqa: E402
from webaggs.truth import grade, load_truth  # noqa: E402

EXP_DIR = config.DATA_DIR / "experiments" / "exp2s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    args = ap.parse_args()

    truth = load_truth(config.DATA_DIR, args.cohort)
    idx = json.loads((EXP_DIR / "runs_index.json").read_text(encoding="utf-8"))
    rows = []
    for player, meta in sorted(idx.items()):
        if not meta.get("done"):
            continue
        ledger = Ledger(EXP_DIR / "runs" / f"{meta['run_id']}.sqlite")
        asserted = {season: value for _rk, season, value, _b, _d
                    in ledger.resolved(meta["run_id"])}
        ledger.close()
        g = grade(player, truth[player], asserted)
        rows.append({"player": player, "certificate": meta["certificate"],
                     "stop_reason": meta["stop_reason"],
                     "n_truth": g.n_truth, "n_asserted": g.n_asserted,
                     "record_recall": round(g.record_recall, 4),
                     "exact_recall": round(g.exact_recall, 4),
                     "n_spurious": g.n_spurious,
                     "true_total": g.true_total,
                     "asserted_total": g.asserted_total,
                     "rel_agg_error": round(g.rel_agg_error, 4)})

    out = EXP_DIR / "results" / "grades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Grades written to {out} ({len(rows)} players).")


if __name__ == "__main__":
    main()
