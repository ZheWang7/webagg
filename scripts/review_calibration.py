"""Review the registry-undecidable calibration mentions (guide Sec. 6.3).

The harvester auto-labels what the truth table can decide; everything
farther than CLAIM_TOL_REL from every filed value lands in a queue for
YOUR judgment, because the registry cannot tell a faithful read of an
out-of-scope figure (IPO proceeds, cumulative totals, full-round sizes)
from a hallucination. This tool shows each mention IN ITS PASSAGE and asks
one question:

    does the passage really say this value?

  y  faithful read  -> the gate learns it as a CORRECT extraction
  n  garbled read   -> labeled an extraction error (paired with the
                       nearest filed value for the distance term)
  s  skip for now   -> stays in the queue
  q  quit           -> everything already answered is saved (each answer
                       writes both files immediately)

Judge READING fidelity only: whether the number matches the sentence, not
whether the sentence matches the world. A page misreporting a round that
the extractor copied faithfully is a 'y' -- the source is wrong, the
reading is right, and source error is corroboration's problem (Sec. 8),
not the gate's.

Usage (repo root, no API keys):

    python scripts/review_calibration.py
    python scripts/review_calibration.py --limit 10      # short session
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                   # noqa: E402
from webagg.calharvest import (record_decision,             # noqa: E402
                               threshold_preview)


def _show(i: int, total: int, row: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{i}/{total}]  {row['entity_id']}  run={row['run_id']}  "
          f"self_conf={row['self_conf']:.2f}")
    passage = (row.get("passage") or "").strip() or "(no passage stored)"
    print(textwrap.fill(passage, width=72, initial_indent="  ",
                        subsequent_indent="  "))
    rel = row.get("rel_dist")
    if rel is None:
        ctx = "(unparseable)"
    elif rel > 10:
        ctx = "(no filed value anywhere near)"
    else:
        ctx = f"(off by {rel:.1%})"
    print(f"\n  extracted : {row['pred']}")
    print(f"  nearest filed value: {row['registry_nearest']}  {ctx}")
    if row.get("url"):
        print(f"  source    : {row['url']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", default=None,
                    help="override the queue path (default: next to the "
                         "calibration file)")
    ap.add_argument("--cal", default=None,
                    help=f"override the calibration file "
                         f"(default: {config.CALIBRATION_SET})")
    ap.add_argument("--limit", type=int, default=None,
                    help="review at most this many, then stop")
    args = ap.parse_args()

    cal_path = Path(args.cal or config.CALIBRATION_SET)
    queue_path = Path(args.queue or cal_path.with_name("review_queue.json"))
    if not queue_path.exists():
        sys.exit("no review queue -- run scripts/build_calibration.py first.")
    queue = json.loads(queue_path.read_text())
    if not queue:
        sys.exit("review queue is empty -- nothing to judge.")

    print(f"{len(queue)} mention(s) await judgment. The question is always: "
          f"does the passage really say this value?  [y/n/s/q]")
    todo = queue[: args.limit] if args.limit else list(queue)
    n_y = n_n = 0
    for i, row in enumerate(todo, 1):
        _show(i, len(todo), row)
        while True:
            ans = input("  faithful read? [y/n/s/q] ").strip().lower()
            if ans in ("y", "n", "s", "q"):
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        record_decision(cal_path, queue_path, row["mention_id"],
                        faithful=(ans == "y"))
        n_y += ans == "y"
        n_n += ans == "n"

    remaining = len(json.loads(queue_path.read_text()))
    print("\n" + "=" * 72)
    print(f"decided this session: {n_y} faithful / {n_n} extraction errors; "
          f"{remaining} still queued")
    if cal_path.exists():
        pv = threshold_preview(cal_path)
        wrong_frac = pv["scores_ge_1"] / max(pv["n"], 1)
        print(f"gate preview: n={pv['n']} decided  "
              f"t_hat={pv['threshold']:.4f}  "
              f"(wrong fraction {wrong_frac:.1%}, delta_E="
              f"{config.DELTA_E:.0%})")


if __name__ == "__main__":
    main()
