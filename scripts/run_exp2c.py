"""Exp 2-C LIVE phase: one run per cohort paper (PREREGISTRATION_EXP2C rev 3).

Each run walks the pre-registered citation indexes one page per capture
occasion (webaggr.pdf Def. 5), with Y_CAP = the page size, and stops by
the unchanged two-conjunct stop test (§3.3). epsilon and tau_max are
REQUIRED flags: they are pre-registered numbers, never defaults.

Usage:
    $env:RUN_LIVE="1"
    python scripts/run_exp2c.py --cohort cite_v1 --eps 0.5 --max-steps 900
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config, design              # noqa: E402
from webaggs.conformal import ConformalGate     # noqa: E402
from webaggs.indexes import Target, make_indexes  # noqa: E402
from webaggs.pipeline import end_to_end         # noqa: E402
from webaggs.search import SerperBackend        # noqa: E402
from webaggs.storage import fresh_ledger        # noqa: E402
from webaggs.truth import load_truth, truth_paths  # noqa: E402

EXP_DIR = config.DATA_DIR / "experiments" / "exp2c"


def load_targets(cohort: str) -> tuple[dict, str]:
    """-> ({paper_norm: Target}, cutoff) written by build_citation_truth."""
    path = truth_paths(config.DATA_DIR, cohort)[0].parent / "targets.json"
    if not path.exists():
        raise SystemExit(f"No {path} — rebuild the cohort with "
                         "scripts/build_citation_truth.py.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ({k: Target(**v) for k, v in data["targets"].items()},
            data["cutoff"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--eps", type=float, required=True,
                    help="pre-registered live epsilon")
    ap.add_argument("--max-steps", type=int, required=True,
                    help="pre-registered tau_max (capture occasions)")
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument("--max-papers", type=int, default=None,
                    help="run at most this many NOT-yet-run papers, then "
                         "stop cleanly (e.g. 1 to check the first run "
                         "before the rest); order is alphabetical, fixed")
    ap.add_argument("--indexes", default="openalex,europepmc",
                    help="comma-separated enumerators (pre-registered)")
    ap.add_argument("--page-size", type=int, default=config.INDEX_PAGE_SIZE)
    ap.add_argument("--index-reader", default=config.INDEX_READER,
                    choices=("structured", "llm"))
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")

    truth = load_truth(config.DATA_DIR, args.cohort)
    targets, cutoff = load_targets(args.cohort)
    gate = ConformalGate.load()
    search = SerperBackend()
    names = [n.strip() for n in args.indexes.split(",") if n.strip()]
    idx_path = EXP_DIR / "runs_index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}

    # the arithmetic, printed before any spend: can the certificate fire?
    need = design.n_needed(args.eps, args.page_size, args.max_steps,
                           config.BETA, config.DELTA_M)
    print(f"design: eps={args.eps} tau_max={args.max_steps} "
          f"Y_CAP={args.page_size} -> N must exceed {need:,.0f} "
          "(more if the indexes overlap imperfectly)")

    ran = 0
    for key in sorted(truth):
        if idx.get(key, {}).get("done"):
            print(f"[skip] {key[:50]} already run")
            continue
        if args.max_papers is not None and ran >= args.max_papers:
            print(f"[stop] --max-papers {args.max_papers} reached; "
                  "re-run the same command to continue")
            break
        ran += 1
        if key not in targets:
            raise SystemExit(f"{key!r} is in truth but not in targets.json")
        tgt = targets[key]
        # same run-id scheme as the notebook (two entry points, one ledger set)
        run_id = f"exp2c_{args.cohort}_{key.replace(' ', '_')[:60]}"
        ledger = fresh_ledger(EXP_DIR / "runs", run_id)
        print(f"[live] {tgt.title[:60]} at eps={args.eps} ...")
        ans = end_to_end(tgt.title, run_id, ledger,
                         search=search, gate=gate, eps_g=args.eps,
                         budget_usd=args.budget, max_steps=args.max_steps,
                         target=tgt, indexes=make_indexes(names),
                         cutoff=cutoff, page_size=args.page_size,
                         y_cap=args.page_size,
                         index_reader=args.index_reader)
        ledger.close()
        idx[key] = {"done": True, "run_id": run_id,
                    "stop_reason": ans.stop_reason,
                    "certificate": ans.certificate,
                    "total": ans.total, "n_records": ans.n_records,
                    "params": {"eps": args.eps, "max_steps": args.max_steps,
                               "page_size": args.page_size,
                               "indexes": names, "cutoff": cutoff,
                               "index_reader": args.index_reader}}
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(idx, indent=2))
        print(f"    -> {ans.stop_reason} / {ans.certificate}: "
              f"{ans.n_records} citing papers found")


if __name__ == "__main__":
    main()
