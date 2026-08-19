"""Certify eps_F by Learn-Then-Test on the withheld-registry cohort (A3).

The run that replaces `eps_F=0.150 [fallback]` in every report with a
CALIBRATED number (guide Sec. 13; paper Theorem 5). Two phases:

  1. REFERENCE RUNS (live, once per calibration entity): full open-web
     discovery with the registry DENIED, one frozen source pool per entity.
     Idempotent -- existing pools are reused, so re-running this script
     after a crash costs nothing.
  2. LTT (replay, cheap-ish): for each lambda in config.LTT_GRID, in the
     PRE-COMMITTED order, replay gate -> ER -> corroboration over every
     frozen pool, score fidelity losses against the build_truth.py answer
     keys, and fixed-sequence test E[L] <= eps_F at level delta_F. Stops at
     the first failing lambda (re-ordering after seeing losses voids the
     guarantee -- guide pitfall 8). The deepest passing lambda is stored as
     data/fidelity_certs/<domain>.json.

Replay needs the OpenAI key (band pairs escalate to the live adjudicator)
and torch (the matcher embeds surfaces). Run where you run live queries.

Usage (repo root):
    python scripts/certify_fidelity.py --cohort formd_v1
    python scripts/certify_fidelity.py --cohort formd_v1 --domain formd_v1 \\
        --eps-f 0.10 --delta-f 0.05 --budget-usd 10 --max-steps 60

After certifying: run_query.py --domain <domain> makes every report read
the certificate; the validation half stays untouched for experiment 8.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.certify import (load_runs_index, make_callables,  # noqa: E402
                            reference_runs)
from webagg.risk_control import (FidelityCertificate,      # noqa: E402
                                 fidelity_loss, learn_then_test,
                                 save_fidelity_cert)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True,
                    help="cohort name under data/ground_truth/ "
                         "(from build_truth.py)")
    ap.add_argument("--domain", default=None,
                    help="certificate domain name (default: the cohort name)")
    ap.add_argument("--eps-f", type=float, default=config.EPS_F_TARGET,
                    help="the fidelity level to ATTEMPT to certify")
    ap.add_argument("--delta-f", type=float, default=config.DELTA_F,
                    help="Learn-Then-Test level (1 - confidence)")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="frontier cap for reference runs")
    ap.add_argument("--budget-usd", type=float, default=config.BUDGET_USD,
                    help="per-entity budget for reference runs")
    ap.add_argument("--refresh", action="store_true",
                    help="force fresh reference runs (default: reuse pools)")
    ap.add_argument("--refresh-entities", nargs="*", default=(),
                    help="refresh ONLY these entity ids (others reuse pools)")
    ap.add_argument("--name", action="append", default=[],
                    metavar="EID=COMMON_NAME",
                    help="open-web query name override, e.g. "
                         "--name cik0001579091=Instacart (the manifest holds "
                         "registry legal names, which the press may not use)")
    ap.add_argument("--amount-tol", type=float,
                    default=config.GRADING_AMOUNT_TOL,
                    help="PRE-REGISTERED grading fallback: undated records "
                         "may align to truth by relative amount distance "
                         "<= this (the cohort's registry_tol); 0 disables")
    args = ap.parse_args()
    domain = args.domain or args.cohort

    cohort_dir = config.GROUND_TRUTH_DIR / args.cohort
    manifest_path = cohort_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"no cohort manifest at {manifest_path} -- run "
                 f"scripts/build_truth.py first.")
    manifest = json.loads(manifest_path.read_text())
    cal_ids = manifest["split"]["calibration"]
    print(f"[certify] cohort={args.cohort}  domain={domain}  "
          f"calibration entities: {cal_ids}")
    print(f"[certify] validation half {manifest['split']['validation']} is "
          f"NOT touched (experiment 8's holdout).")

    # ---- phase 1: the frozen pools ---------------------------------------
    names = dict(kv.split("=", 1) for kv in args.name)
    index = reference_runs(manifest, domain=domain,
                           max_steps=args.max_steps,
                           budget_usd=args.budget_usd,
                           refresh=args.refresh, names=names,
                           refresh_entities=tuple(args.refresh_entities))
    missing = [e for e in cal_ids if e not in index]
    if missing:
        sys.exit(f"reference runs missing for {missing}; aborting.")

    # ---- phase 2: fixed-sequence LTT over the pre-committed grid ---------
    run_pipeline, truth = make_callables(index, cohort_dir)
    trace: list = []
    tol = args.amount_tol if args.amount_tol > 0 else None
    loss_fn = functools.partial(fidelity_loss, amount_tol=tol)
    certified = learn_then_test(cal_ids, config.LTT_GRID,
                                args.eps_f, args.delta_f,
                                run_pipeline=run_pipeline, truth=truth,
                                loss_fn=loss_fn, trace=trace)

    print(f"\n{'lambda':<58}{'mean L':>8}{'p':>10}  verdict")
    for t in trace:
        print(f"{str(t['lam']):<58}{t['mean_loss']:>8.4f}{t['p']:>10.4f}  "
              f"{'PASS' if t['passed'] else 'FAIL (stop)'}")

    if certified is None:
        print(f"\nNOT CERTIFIED: the first lambda already failed "
              f"E[L] <= {args.eps_f} at delta_F={args.delta_f}. The App. H "
              f"fallback ({config.EPS_F_FALLBACK}) remains in force. Honest "
              f"options: inspect the per-entity losses (the trace above), "
              f"grow the cohort, or attempt a LOOSER eps-f target -- but "
              f"only as a NEW pre-registered attempt, not a re-roll.")
        sys.exit(1)

    lam_star, mean_loss = certified
    cert = FidelityCertificate(domain=domain, eps_F=args.eps_f,
                               delta_F=args.delta_f, method="ltt",
                               lam=lam_star, mean_loss=mean_loss,
                               n_cal=len(cal_ids),
                               grading={"amount_tol": tol,
                                        "key": "base_kind|date, "
                                               "amount-tol fallback"})
    path = save_fidelity_cert(cert)
    print(f"\nCERTIFIED: eps_F={args.eps_f} at 1-delta_F="
          f"{1 - args.delta_f:.2f}  (lambda*={lam_star}, realized mean "
          f"L={mean_loss:.4f}, n={len(cal_ids)})")
    print(f"certificate: {path}")
    print(f"\nnext: run_query.py --domain {domain} now reads this "
          f"certificate; experiment 8 audits it on the validation half.")


if __name__ == "__main__":
    main()
