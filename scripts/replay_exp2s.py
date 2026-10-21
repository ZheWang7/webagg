"""Exp 2-S OFFLINE phase: replay looser-epsilon arms from the frozen ledgers
and emit the C1/C2 tables. No fetches, no LLM calls, no spend.

Self-tests (run before any table is written; a failure aborts loudly):
  S1  replay at the LIVE epsilon reproduces the live certified stop step
      for every run that stopped 'certified';
  S2  replayed stop steps are monotone non-increasing as epsilon loosens.

Outputs (data/experiments/exp2s/results/):
  stops.csv        one row per (player, eps): stop step, stop kind
                   (statistical/checksum/censored), spend at stop,
                   realized record recall at stop   -> C2 / Plot 2
  calibration.csv  one row per (player, step): claimed unseen (U_hat + psi)
                   vs realized unseen (1 - recall)  -> C1 / Plot 1; points
                   ABOVE the diagonal falsify Assumption 1(a)
  shortfall.csv    per (player, eps): truth seasons missed, split into
                   lost-to-stopping-early (found later in the recorded
                   trajectory) vs never-discovered  -> the Fig-5 analogue

Reporting split (pre-registered): checksum-path stops are flagged in
stop_kind and EXCLUDED from the statistical C1/C2 headline counts printed
at the end.

No pandas anywhere in this script — the None->NaN truthiness trap that bit
exp3/exp7 cannot recur if the columns never pass through an object dtype.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402
from webaggs.replay import (recall_at_step, recall_trajectory,  # noqa: E402
                            replay_stop)
from webaggs.storage import Ledger              # noqa: E402
from webaggs.truth import load_truth            # noqa: E402

EXP_DIR = config.DATA_DIR / "experiments" / "exp2s"
EPS_ARMS = (0.30, 0.20, 0.10, 0.05)     # 0.05 = the live arm
LIVE_EPS = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    args = ap.parse_args()

    truth = load_truth(config.DATA_DIR, args.cohort)
    idx = json.loads((EXP_DIR / "runs_index.json").read_text(encoding="utf-8"))
    results_dir = EXP_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    stops_rows, cal_rows, short_rows = [], [], []
    stat_stops = stat_violations = 0

    for player, meta in sorted(idx.items()):
        if not meta.get("done"):
            continue
        ledger = Ledger(EXP_DIR / "runs" / f"{meta['run_id']}.sqlite")
        run_id = meta["run_id"]
        ms = ledger.measurements(run_id, metric="U_hat")
        cov = ledger.coverage(run_id)
        live_stop = ledger.stop(run_id)
        ledger.close()
        truth_keys = {f"{player}|{season}" for season in truth[player]}

        # ---- self-test S1: replay at live eps reproduces the live stop ----
        if live_stop and live_stop[1] == "certified":
            rr = replay_stop(ms, LIVE_EPS)
            assert rr.stop_step == live_stop[0], (
                f"S1 FAILED for {player}: live certified stop at step "
                f"{live_stop[0]}, replay at eps={LIVE_EPS} says {rr.stop_step}. "
                "The ledger does not reproduce the live decision — do not "
                "report any replayed arm until this is diagnosed.")

        # ---- replay every arm + self-test S2 (monotonicity) ---------------
        prev_step = None
        for eps in sorted(EPS_ARMS):            # strictest first: 0.05 ... 0.30
            rr = replay_stop(ms, eps)
            if prev_step is not None:
                # a stricter eps stopped -> every looser eps must stop, no later
                assert rr.stop_step is not None and rr.stop_step <= prev_step, (
                    f"S2 FAILED for {player}: eps={eps} stops at "
                    f"{rr.stop_step} but a stricter eps stopped at {prev_step} "
                    "— the stop test is not monotone in eps.")
            prev_step = rr.stop_step if rr.stop_step is not None else prev_step

            recall = recall_at_step(cov, truth_keys, rr.stop_step)
            # stop kind: checksum if the stratum was checksum-certified at the
            # stop step (read from the logged extras — "" means not certified)
            kind = "censored"
            if rr.stop_step is not None:
                row = next(r for r in ms if r[0] == rr.stop_step)
                kind = "checksum" if str(row[4].get("certified") or "") \
                    else "statistical"
                spend = row[4].get("spent", 0.0)
            else:
                spend = ms[-1][4].get("spent", 0.0) if ms else 0.0
            stops_rows.append({"player": player, "eps": eps,
                               "stop_step": rr.stop_step if rr.stop_step is not None else "",
                               "stop_kind": kind, "censored": int(rr.censored),
                               "spend_at_stop": spend,
                               "recall_at_stop": round(recall, 4),
                               "promised": 1 - eps,
                               "violated": int(recall < 1 - eps)})
            if eps == LIVE_EPS and kind == "statistical":
                stat_stops += 1
                stat_violations += int(recall < 1 - eps)

            # ---- shortfall decomposition (Fig-5 analogue) ------------------
            found_by_stop = {rk for s, rk in cov if rk in truth_keys
                             and (rr.stop_step is None or s <= rr.stop_step)}
            found_ever = {rk for _, rk in cov if rk in truth_keys}
            short_rows.append({
                "player": player, "eps": eps,
                "n_truth": len(truth_keys),
                "lost_to_stopping": len(found_ever - found_by_stop),
                "never_discovered": len(truth_keys - found_ever)})

        # ---- C1 calibration points: claimed vs realized unseen, every step -
        recall_traj = dict(recall_trajectory(cov, truth_keys))
        running = 0.0
        for step, _g, _m, u, extra in ms:
            running = recall_traj.get(step, running)
            cal_rows.append({"player": player, "step": step,
                             "claimed_unseen": round(min(u + extra["psi"], 1.0), 4),
                             "raw_U_hat": round(u, 4),
                             "realized_unseen": round(1 - running, 4)})

    _write(results_dir / "stops.csv", stops_rows)
    _write(results_dir / "calibration.csv", cal_rows)
    _write(results_dir / "shortfall.csv", short_rows)

    above = sum(1 for r in cal_rows
                if r["claimed_unseen"] < r["realized_unseen"])
    print(f"Self-tests S1/S2 passed over {len(idx)} runs.")
    print(f"C1: {above}/{len(cal_rows)} certificate points above the diagonal "
          "(claimed < realized) — any such point falsifies Assumption 1(a).")
    print(f"C2 (statistical path only, eps={LIVE_EPS}): {stat_stops} certified "
          f"stops, {stat_violations} recall violations "
          f"(delta_M budget = {0.05}).")
    print(f"Tables written to {results_dir}")


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
