"""Exp 2-C PILOT: instrument calibration BEFORE the pre-registration locks.

One live run on a PILOT paper (never in the cohort) measures the two
unknowns the prereg needs:

  1. Y_CAP — the real per-occasion novelty bound (Assumption 1(b)): we log
     new records per capture occasion and report the max and the implied
     epsilon ladder  n_needed(eps) = Y_CAP*(1+beta)*sqrt(2*tau_max*
     ln(1/delta_M))/eps  so the live epsilon is chosen from measurement;
  2. the conformal calibration rows for the CITATION extractor (titles/
     DOIs are a new reading task; the NHL gate does not transfer) — dumped
     with an empty `correct` column for hand-labeling against the pilot
     paper's public citation listings.

Usage:
    $env:RUN_LIVE="1"
    python scripts/pilot_ycap.py --paper "Attention Is All You Need" --budget 3.0
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402
from webaggs.conformal import ConformalGate     # noqa: E402
from webaggs.storage import fresh_ledger        # noqa: E402

OUT = config.DATA_DIR / "pilot_exp2c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True,
                    help="pilot paper TITLE — must NOT be in the cohort")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=40)
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")

    from webaggs.canon import normalize_title   # noqa: E402
    from webaggs.pipeline import run_query      # noqa: E402
    from webaggs.search import SerperBackend    # noqa: E402

    gate = ConformalGate(threshold=1.0, n_cal=0)   # permissive: label later
    rid = "pilot_" + normalize_title(args.paper).replace(" ", "_")[:40]
    ledger = fresh_ledger(OUT / "runs", rid)
    print(f"[live] pilot on {args.paper!r} (budget ${args.budget}) ...")
    run_query(args.paper, rid, ledger, search=SerperBackend(), gate=gate,
              budget_usd=args.budget, max_steps=args.max_steps)

    # --- Y_CAP measurement: NEW records per capture occasion -------------
    per_step = Counter(s for s, _ in ledger.coverage(rid))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "novelty_per_occasion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "new_records"])
        for s in sorted(per_step):
            w.writerow([s, per_step[s]])
    ycap = max(per_step.values()) if per_step else 0
    n_total = sum(per_step.values())
    coeff = (ycap * (1 + config.BETA)
             * math.sqrt(2 * config.MAX_STEPS * math.log(1 / config.DELTA_M)))
    print(f"\nOccasions with discoveries: {len(per_step)}; records: {n_total}")
    print(f"MEASURED max novelty/occasion (candidate Y_CAP): {ycap}")
    print("Implied ladder  n_needed(eps) = Y_CAP(1+beta)*sqrt(2*tau*ln(1/dM))/eps:")
    for eps in (0.5, 0.4, 0.3, 0.2, 0.1):
        print(f"  eps={eps}: n >= {coeff / eps:,.0f}")
    print("-> pre-register Y_CAP and the live eps from this table, then "
          "pick cohort papers whose citer counts clear the chosen n.")

    # --- conformal calibration rows for hand-labeling --------------------
    rows = ledger.conn.execute(
        "SELECT m.confidence, m.season, m.entity_surface, s.url "
        "FROM mentions m LEFT JOIN sources s ON s.source_id=m.source_id "
        "WHERE m.accepted=1 ORDER BY m.confidence DESC").fetchall()
    cal = OUT / "calibration_to_label.csv"
    with open(cal, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "correct", "cite_key", "cited_paper",
                    "source_url"])
        for conf, key, surf, url in rows:
            w.writerow([conf, "", key, surf, url or ""])
    ledger.close()
    print(f"\n{len(rows)} extraction rows -> {cal}")
    print("Label `correct` (1 = really cites the pilot paper, checked "
          "against its public cited-by listing), save as "
          "data/conformal/calibration.csv (>= 20 correct rows).")


if __name__ == "__main__":
    main()
