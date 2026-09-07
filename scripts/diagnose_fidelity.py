"""Diagnose a failed fidelity certification: WHERE did the loss come from?

Replays the pre-committed grid's FIRST lambda (the one the LTT trace
scored) over the frozen reference pools and breaks each entity's loss into
its two channels:

  * amount error on MATCHED records  (right round, wrong number)
  * SPURIOUS records                 (no true counterpart: wrong-company
                                      contamination, aggregator totals,
                                      double counting -- full price)

Missing rounds are completeness, not fidelity, so an entity that asserted
nothing scores 0.0 -- the mean cannot be blamed on starvation.

Read-only with respect to the pools and the truth tables; no new
discovery. Live cost: only band-pair escalations to the ER adjudicator,
same as one replay row. The validation half is never touched.

Usage (repo root):
    python scripts/diagnose_fidelity.py --cohort formd_v2 --amount-primary
    python scripts/diagnose_fidelity.py --cohort formd_v2 --amount-primary --top 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.certify import load_runs_index, replay         # noqa: E402
from webagg.formd import load_truth_cohort                 # noqa: E402
from webagg.risk_control import (_attrs, match_to_truth,   # noqa: E402
                                 usd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--amount-tol", type=float,
                    default=config.GRADING_AMOUNT_TOL)
    ap.add_argument("--amount-primary", action="store_true")
    ap.add_argument("--top", type=int, default=8,
                    help="dump full record-vs-truth detail for the N worst")
    args = ap.parse_args()
    domain = args.domain or args.cohort

    cohort_dir = config.GROUND_TRUTH_DIR / args.cohort
    manifest = json.loads((cohort_dir / "manifest.json").read_text())
    truths = load_truth_cohort(cohort_dir)
    index = load_runs_index(domain)
    lam = config.LTT_GRID[0]                 # the lambda the trace scored
    tol = args.amount_tol if args.amount_tol > 0 else None
    cache: dict = {}

    rows = []
    detail: dict[str, list] = {}
    for eid in manifest["split"]["calibration"]:
        info = index.get(eid)
        if info is None or not Path(info["db"]).exists():
            print(f"!! no frozen pool for {eid} -- skipped"); continue
        recs = replay(info["db"], lam, delta_E_ref=info["delta_E_ref"],
                      matcher_cache=cache)
        truth = truths[eid]
        aligned = match_to_truth(recs, truth, amount_tol=tol,
                                 amount_primary=args.amount_primary)
        assembled = sum(usd(_attrs(r).get("amount"))
                        for r, t in aligned if t is not None)
        true_match = sum(t.amount for r, t in aligned if t is not None)
        spurious = sum(usd(_attrs(r).get("amount"))
                       for r, t in aligned if t is None)
        err = abs(assembled - true_match) + spurious
        loss = min(1.0, err / max(truth.true_sum, 1e-9))
        rows.append({
            "eid": eid, "name": info["entity_name"], "loss": loss,
            "n_rec": len(aligned),
            "n_match": sum(1 for _, t in aligned if t is not None),
            "n_spur": sum(1 for _, t in aligned if t is None),
            "amount_err": abs(assembled - true_match),
            "spurious": spurious, "true_sum": truth.true_sum,
        })
        detail[eid] = aligned

    rows.sort(key=lambda r: -r["loss"])
    M = lambda v: f"{v/1e6:,.1f}"           # noqa: E731  (millions)
    print(f"\n{'entity':<30}{'loss':>7}{'rec':>5}{'match':>6}{'spur':>6}"
          f"{'amt err $M':>12}{'spurious $M':>13}{'true $M':>10}")
    for r in rows:
        print(f"{r['name'][:29]:<30}{r['loss']:>7.3f}{r['n_rec']:>5}"
              f"{r['n_match']:>6}{r['n_spur']:>6}{M(r['amount_err']):>12}"
              f"{M(r['spurious']):>13}{M(r['true_sum']):>10}")

    n = len(rows) or 1
    tot_spur = sum(r["spurious"] / max(r["true_sum"], 1e-9) for r in rows)
    tot_amt = sum(min(1.0, r["amount_err"] / max(r["true_sum"], 1e-9))
                  for r in rows)
    print(f"\nmean loss          : {sum(r['loss'] for r in rows)/n:.4f}")
    print(f"entities at cap 1.0: {sum(1 for r in rows if r['loss'] >= 0.999)}")
    print(f"asserted nothing   : {sum(1 for r in rows if r['n_rec'] == 0)}"
          f"  (these score 0.0 -- fidelity does not price missing)")
    print(f"loss share, spurious channel   ~ {tot_spur/n:.3f}")
    print(f"loss share, amount-error chan. ~ {tot_amt/n:.3f}")

    print(f"\n=== the {args.top} worst, record by record ===")
    for r in rows[:args.top]:
        print(f"\n-- {r['name']}  (loss {r['loss']:.3f}, "
              f"true sum ${M(r['true_sum'])}M)")
        for rec, t in detail[r["eid"]]:
            a = _attrs(rec)
            amt = usd(a.get("amount"))
            date = getattr(a.get("date"), "value", None)
            who = rec.get("entity_id", "?") if isinstance(rec, dict) \
                else getattr(rec, "entity_id", "?")
            if t is None:
                print(f"   SPURIOUS  ${M(amt):>9}M  date={date!s:<12} "
                      f"resolved-entity={who!r}")
            else:
                print(f"   matched   ${M(amt):>9}M  vs true ${M(t.amount):>9}M"
                      f"  ({t.key})  resolved-entity={who!r}")


if __name__ == "__main__":
    main()
