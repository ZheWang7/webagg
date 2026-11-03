"""Collect conformal-gate calibration rows from ONE pilot player (live spend).

The gate (webaggs/conformal.py) needs labeled (confidence, correct) pairs
before any experiment run — it refuses cold starts. This script produces the
rows to label:

  1. runs REAL discovery + extraction on a pilot player with a PERMISSIVE
     gate (threshold 1.0: nothing is filtered, so the labeled sample covers
     the whole confidence range — fitting the gate on pre-filtered rows
     would bias the quantile);
  2. dumps every typed-valid extracted row to data/conformal/calibration.csv
     with an EMPTY `correct` column.

You then fill `correct` (1 if the row matches the player's official page on
nhl.com — which you check in a BROWSER; this script never touches the
oracle, and the pilot stays out of the cohort so no experiment data is
exposed) and save. The gate loader reads only the confidence/correct
columns; the extra context columns are for your labeling convenience.

Pre-registration discipline: the pilot player must NOT be in the cohort,
and calibration happens BEFORE the cohort freeze is used for any live run.

Usage:
    $env:RUN_LIVE="1"
    python scripts/collect_calibration.py --player "Zach Aston-Reese" --budget 1.0
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402
from webaggs.conformal import ConformalGate     # noqa: E402
from webaggs.storage import Ledger, fresh_ledger  # noqa: E402

OUT_PATH = config.DATA_DIR / "conformal" / "calibration.csv"


def dump_rows(ledger: Ledger, run_id: str, out_path: Path) -> int:
    """Write every accepted-at-permissive-gate mention to the labeling CSV.
    Columns: confidence, correct (EMPTY — you fill it), then context."""
    rows = ledger.conn.execute(
        "SELECT m.confidence, m.entity_surface, m.season, m.value, s.url "
        "FROM mentions m LEFT JOIN sources s ON s.source_id = m.source_id "
        "WHERE m.accepted = 1 ORDER BY m.season, m.confidence DESC").fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "correct", "player_surface", "season",
                    "points", "source_url"])
        for conf, surface, season, value, url in rows:
            w.writerow([conf, "", surface, season, value, url or ""])
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", required=True,
                    help="pilot player's display name — must NOT be in the cohort")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="search budget in $ (default 1.0 — a pilot, not a run)")
    ap.add_argument("--max-steps", type=int, default=25)
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend. This script "
                         "runs real searches and LLM extraction.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")
    if OUT_PATH.exists():
        raise SystemExit(
            f"{OUT_PATH} already exists. A fitted calibration file must not "
            "be silently overwritten — move it aside first if you really "
            "mean to re-collect (that is an instrument change: note it).")

    from webaggs.canon import normalize_name   # noqa: E402  (after gates)
    from webaggs.pipeline import run_query     # noqa: E402
    from webaggs.search import SerperBackend   # noqa: E402

    # permissive gate: accept EVERYTHING so labels cover the whole range
    gate = ConformalGate(threshold=1.0, n_cal=0)
    run_id = f"calib_{normalize_name(args.player).replace(' ', '_')}"
    ledger = fresh_ledger(config.DATA_DIR / "conformal" / "runs", run_id)
    print(f"[live] pilot discovery for {args.player!r} "
          f"(budget ${args.budget}, permissive gate) ...")
    run_query(args.player, run_id, ledger, search=SerperBackend(), gate=gate,
              budget_usd=args.budget, max_steps=args.max_steps)
    n = dump_rows(ledger, run_id, OUT_PATH)
    ledger.close()

    print(f"\n{n} rows written to {OUT_PATH}")
    print("Next: open it, fill the `correct` column (1 = matches the "
          "player's official nhl.com page, 0 = wrong), save. The gate needs "
          ">= 20 rows labeled CORRECT; if this pilot yielded fewer rows, "
          "re-run with a second pilot player and append.")


if __name__ == "__main__":
    main()
