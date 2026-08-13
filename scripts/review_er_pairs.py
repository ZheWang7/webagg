"""Label queued ER pairs: same entity or different? (seam A2, step 2)

Walks the unlabeled rows of data/ground_truth/match_pairs.csv and records
your verdicts. The question is always the operational one -- SHOULD a
correct join place these two mentions in the same cluster? With
instance-qualified record kinds ("funding_round/seed") that means: same
company AND same round-stage grouping. The kinds are shown; when they
differ, the answer is almost always n.

Keys (same convention as review_calibration.py):
    y = same (label 1)      n = different (label 0)
    s = skip for now        q = quit (progress is saved per answer)

The bucket shown ("easy_pos"/"easy_neg"/"band") is where the SAMPLER put
the pair, not the answer -- disagreeing with it is exactly the informative
case, so judge the evidence, not the bucket.

Usage (repo root):
    python scripts/review_er_pairs.py
    python scripts/review_er_pairs.py --limit 25
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.er_pairs import labeled_matrix, load_rows, write_rows  # noqa: E402


def _show(i: int, total: int, r: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{i}/{total}]  bucket={r['bucket']}  theta_cold={r['theta_cold']}")
    print(f"  A: {r['surface_a']!r}   kind={r['kind_a']}   "
          f"domain={r['domain_a']}")
    if r["passage_a"]:
        print(f"     \"{r['passage_a']}\"")
    print(f"  B: {r['surface_b']!r}   kind={r['kind_b']}   "
          f"domain={r['domain_b']}")
    if r["passage_b"]:
        print(f"     \"{r['passage_b']}\"")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=None,
                    help=f"pairs CSV (default: {config.MATCH_PAIRS})")
    ap.add_argument("--limit", type=int, default=None,
                    help="review at most this many, then stop")
    args = ap.parse_args()

    path = Path(args.pairs or config.MATCH_PAIRS)
    rows = load_rows(path)
    if not rows:
        sys.exit("no pairs file -- run scripts/build_er_pairs.py first.")
    todo = [r for r in rows if not (r["label"] or "").strip()]
    if not todo:
        sys.exit("nothing awaiting labels.")
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(todo)} pair(s) await judgment. Same cluster under a "
          f"correct join?  [y/n/s/q]")
    n_same = n_diff = 0
    for i, r in enumerate(todo, 1):
        _show(i, len(todo), r)
        while True:
            ans = input("  same? [y/n/s/q] ").strip().lower()
            if ans in ("y", "n", "s", "q"):
                break
        if ans == "q":
            break
        if ans == "s":
            continue
        r["label"] = "1" if ans == "y" else "0"   # rows are the SAME dicts
        write_rows(path, rows)                    # save after EVERY answer --
        n_same += ans == "y"                      # a crash loses nothing
        n_diff += ans == "n"

    X, y = labeled_matrix(load_rows(path))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    remaining = sum(1 for r in load_rows(path)
                    if not (r["label"] or "").strip())
    print("\n" + "=" * 72)
    print(f"decided this session: {n_same} same / {n_diff} different; "
          f"{remaining} still queued")
    print(f"labeled overall: {len(y)} ({n_pos} same / {n_neg} different); "
          f"fitting floor is {config.ER_MIN_LABELED} with "
          f">={config.ER_MIN_PER_CLASS} per class")
    if (len(y) >= config.ER_MIN_LABELED
            and min(n_pos, n_neg) >= config.ER_MIN_PER_CLASS):
        print("floor MET -- the next pipeline run fits the matcher and "
              "prints its alpha.")


if __name__ == "__main__":
    main()
