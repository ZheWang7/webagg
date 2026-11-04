"""Exp 2-C LIVE phase: one run per cohort paper at the pre-registered live
epsilon (chosen from the pilot's measured ladder; see PREREGISTRATION_EXP2C).

Usage:
    $env:RUN_LIVE="1"
    python scripts/run_exp2c.py --cohort cite_v1 --eps 0.30 --budget 20
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

EXP_DIR = config.DATA_DIR / "experiments" / "exp2c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--eps", type=float, required=True,
                    help="pre-registered live epsilon (from the pilot ladder)")
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")

    truth = load_truth(config.DATA_DIR, args.cohort)
    display = load_display_names(config.DATA_DIR, args.cohort)
    gate = ConformalGate.load()
    search = SerperBackend()
    idx_path = EXP_DIR / "runs_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    for key in sorted(truth):
        if idx.get(key, {}).get("done"):
            print(f"[skip] {key[:50]} already run")
            continue
        run_id = f"exp2c_{args.cohort}_{abs(hash(key)) % 10**8}"
        ledger = fresh_ledger(EXP_DIR / "runs", run_id)
        print(f"[live] {key[:60]} at eps={args.eps} ...")
        ans = end_to_end(display.get(key, key), run_id, ledger,
                         search=search, gate=gate, eps_g=args.eps,
                         budget_usd=args.budget, max_steps=args.max_steps)
        ledger.close()
        idx[key] = {"done": True, "run_id": run_id,
                    "stop_reason": ans.stop_reason,
                    "certificate": ans.certificate,
                    "total": ans.total, "n_records": ans.n_records}
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(idx, indent=2))
        print(f"    -> {ans.stop_reason} / {ans.certificate}: "
              f"{ans.n_records} citing papers found")


if __name__ == "__main__":
    main()
