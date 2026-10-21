"""Exp 2-S LIVE phase: one run per cohort player at the strict live epsilon.

Design (mirrors the finance Exp 2 full-cohort notebook): LIVE at eps=0.05,
looser arms {0.30, 0.20, 0.10} replayed OFFLINE from the recorded ledgers by
scripts/replay_exp2s.py — valid because eps is read only in the stop test.

Guards:
  * RUN_LIVE=1 plus both API keys required, else the script refuses (pytest
    suite must be green before any live spend — the gate prints the reminder).
  * Idempotent per-player runs index: a finished player is never re-rolled.
  * The oracle deny list is enforced inside fetch_url for every request.

Usage:
    set RUN_LIVE=1   (PowerShell: $env:RUN_LIVE="1")
    python scripts/run_exp2s.py --cohort nba_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402
from webaggs.conformal import ConformalGate     # noqa: E402
from webaggs.pipeline import end_to_end         # noqa: E402
from webaggs.search import SerperBackend        # noqa: E402
from webaggs.storage import fresh_ledger        # noqa: E402
from webaggs.truth import load_display_names, load_truth  # noqa: E402

EXP_DIR = config.DATA_DIR / "experiments" / "exp2s"
LIVE_EPS = 0.05        # pre-registered live arm (PREREGISTRATION_EXP2S.md)
LIVE_DELTA_M = 0.05


def runs_index() -> dict:
    p = EXP_DIR / "runs_index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_index(idx: dict) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "runs_index.json").write_text(json.dumps(idx, indent=2),
                                             encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--budget", type=float, default=config.BUDGET_USD)
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend. Run the pytest "
                         "suite first, then set RUN_LIVE=1.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")

    truth = load_truth(config.DATA_DIR, args.cohort)   # verifies freeze hash
    display = load_display_names(config.DATA_DIR, args.cohort)
    gate = ConformalGate.load()                        # refuses cold starts
    search = SerperBackend()
    idx = runs_index()

    for player_norm in sorted(truth):
        if idx.get(player_norm, {}).get("done"):
            print(f"[skip] {player_norm} already run")
            continue
        run_id = f"exp2s_{args.cohort}_{player_norm.replace(' ', '_')}"
        # fresh_ledger moves any unfinished (crashed) ledger aside — never
        # append a second trajectory under the same run_id (replay corruption)
        ledger = fresh_ledger(EXP_DIR / "runs", run_id)
        print(f"[live] {player_norm} at eps={LIVE_EPS} ...")
        ans = end_to_end(display.get(player_norm, player_norm), run_id, ledger,
                         search=search, gate=gate,
                         eps_g=LIVE_EPS, delta_m=LIVE_DELTA_M,
                         budget_usd=args.budget)
        ledger.close()
        idx[player_norm] = {"done": True, "run_id": run_id,
                            "stop_reason": ans.stop_reason,
                            "certificate": ans.certificate,
                            "total": ans.total, "n_records": ans.n_records}
        save_index(idx)                                 # idempotent resume point
        print(f"    -> {ans.stop_reason} / {ans.certificate}: "
              f"{ans.n_records} records, total {ans.total}")

    print("\nLive phase complete. Next: python scripts/replay_exp2s.py "
          f"--cohort {args.cohort}")


if __name__ == "__main__":
    main()
