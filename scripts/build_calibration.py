"""Build the conformal-gate calibration set from a denied open-web run
(impl guide Sec. 6.3 data, harvested via the Sec. 15 truth table).

Closes seam A1. Usage (repo root; no API keys needed -- both DBs are local):

    # 1. one live open-web run per CALIBRATION-half entity, experiment
    #    conditions (this part needs your keys):
    python scripts/run_query.py "SpaceX funding rounds total raised" \
           --deny sec --budget-usd 0.50 --run-id cal_spacex_01

    # 2. label that run's amount mentions against the cohort truth table:
    python scripts/build_calibration.py --cohort formd_v1 \
           --run cal_spacex_01 --entity cik0001181412 \
           --aliases SpaceX "Space Exploration Technologies"

Repeat step 1+2 per calibration entity / per run; rows append idempotently
(dedupe by mention_id). Once data/calibration/extraction_cal.json exists,
run_query fits the gate automatically at the next run start -- the
gate_uncalibrated stamp disappears and delta_E means what it says.

The three refusals (see webagg/calharvest.py for why each exists):
circularity (gated runs), split leakage (validation entities), and
contamination (undenied runs / registry pages in the artifact).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                   # noqa: E402
from webagg.calharvest import (MIN_RECOMMENDED_N,           # noqa: E402
                               append_calibration, check_run_conditions,
                               check_split, harvest, threshold_preview)
from webagg.storage import get_session                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True,
                    help="cohort name under data/ground_truth/")
    ap.add_argument("--run", required=True,
                    help="run_id of the DENIED open-web harvest run")
    ap.add_argument("--entity", required=True,
                    help="entity id from the cohort manifest, e.g. cik0001181412")
    ap.add_argument("--aliases", nargs="*", default=[],
                    help="surface forms of the entity as the open web spells "
                         "them (the filed entityName is always included)")
    ap.add_argument("--truth-db", default=None,
                    help="override the truth DB path (default: manifest's)")
    ap.add_argument("--out", default=None,
                    help=f"override output (default: {config.CALIBRATION_SET})")
    ap.add_argument("--allow-undenied", action="store_true",
                    help="accept a harvest run without --deny sec (stamped "
                         "into the manifest; use only for exploration)")
    args = ap.parse_args()

    cohort_dir = config.GROUND_TRUTH_DIR / args.cohort
    manifest = json.loads((cohort_dir / "manifest.json").read_text())
    check_split(manifest, args.entity)                    # guard 2, first

    entity_meta = json.loads(
        (cohort_dir / f"truth_{args.entity}.json").read_text())
    entity_name = entity_meta["entity_name"]
    aliases = [entity_name] + list(args.aliases)

    run_session = get_session(f"data/runs/{args.run}.sqlite")
    truth_session = get_session(args.truth_db or manifest["truth_db"])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_run_conditions(run_session,             # guard 3
                                 allow_undenied=args.allow_undenied)
        rows, stats = harvest(run_session, truth_session,  # guard 1 inside
                              run_id=args.run, entity_id=args.entity,
                              entity_name=entity_name, aliases=aliases)
    finally:
        run_session.close()
        truth_session.close()

    out_path = Path(args.out or config.CALIBRATION_SET)
    n_added, n_total = append_calibration(out_path, rows)

    # -- sidecar manifest: every harvest leaves its provenance ---------------
    man_path = out_path.with_suffix(".manifest.json")
    man = json.loads(man_path.read_text()) if man_path.exists() else {
        "calibration_file": str(out_path), "harvests": []}
    man["harvests"].append({
        "at": datetime.utcnow().isoformat(),
        "cohort": args.cohort, "run_id": args.run, "entity_id": args.entity,
        "aliases": aliases, "stats": stats, "n_added": n_added,
        "undenied_accepted": bool(args.allow_undenied and caught),
    })
    man["n_total"] = n_total
    man["n_below_recommended"] = n_total < MIN_RECOMMENDED_N
    man_path.write_text(json.dumps(man, indent=2))

    # -- the receipt ---------------------------------------------------------
    pv = threshold_preview(out_path)
    print(f"harvested  : {stats['harvested']} amount mentions "
          f"({stats['out_of_scope']} out-of-scope surfaces skipped)")
    print(f"labels     : {stats['correct']} correct / {stats['wrong']} wrong "
          f"(of which {stats['wrong_within_tol']} within "
          f"{config.CLAIM_TOL_REL:.0%} = rounding-style)")
    print(f"file       : {out_path}  (+{n_added} new, {n_total} total)")
    # HONEST accept-all disambiguation: t_hat >= 1 can mean two OPPOSITE
    # things -- the set is so clean that accepting everything satisfies
    # delta_E (fine), or the wrong-label mass EXCEEDS delta_E and the
    # deployment score (1 - self_conf <= 1) simply cannot reach the
    # threshold (the gate has no lever; the label mix is the problem).
    wrong_frac = pv["scores_ge_1"] / max(pv["n"], 1)
    if not pv["accepts_everything"]:
        verdict = (f"rejects mentions with self_conf < "
                   f"{pv['min_self_conf_accepted']:.3f}")
    elif wrong_frac <= config.DELTA_E:
        verdict = (f"calibrated ACCEPT-ALL (wrong fraction "
                   f"{wrong_frac:.1%} <= delta_E={config.DELTA_E:.0%}: "
                   f"accepting everything already meets the guarantee)")
    else:
        verdict = (f"ACCEPT-ALL BY DEFAULT: wrong fraction {wrong_frac:.1%} "
                   f"> delta_E={config.DELTA_E:.0%}, and deployment scores "
                   f"(1 - self_conf) cannot exceed 1 -- self_conf does not "
                   f"separate right from wrong here. More/cleaner examples "
                   f"needed before delta_E is meaningful.")
    print(f"gate preview: n={pv['n']}  t_hat={pv['threshold']:.4f}  ->  "
          + verdict)
    if n_total < MIN_RECOMMENDED_N:
        print(f"\n!! only {n_total} examples -- the guide asks for "
              f"~{MIN_RECOMMENDED_N}+. Harvest more calibration entities / "
              f"runs before treating delta_E as meaningful.")


if __name__ == "__main__":
    main()
