"""Sample ER pairs from a run DB into the labeling queue (seam A2, step 1).

Draws a stratified queue of candidate mention pairs -- easy positives, easy
negatives, and (oversampled) escalation-band hard cases, per the guide's
labeling recipe -- from a real run's accepted mentions, and merges it into
data/ground_truth/match_pairs.csv (existing rows and their labels always
survive; re-runs dedupe and report an honest '+N new').

Needs sentence-transformers/torch (features include the embedding cosine):
run on the machine you do live runs on. No LLM, no network.

Usage (repo root):
    python scripts/build_er_pairs.py --db data/runs/<run_id>.sqlite
    python scripts/build_er_pairs.py --db data/runs/a.sqlite data/runs/b.sqlite \\
        --n-band 100 --seed 0

Then label with scripts/review_er_pairs.py, and the next pipeline run picks
the fitted matcher up automatically (alpha printed at startup).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.er_pairs import append_rows, load_rows, sample_pairs  # noqa: E402
from webagg.storage import get_session, load_mentions, load_sources  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", nargs="+", required=True,
                    help="run database(s) to sample pairs from")
    ap.add_argument("--out", default=None,
                    help=f"pairs CSV (default: {config.MATCH_PAIRS})")
    ap.add_argument("--n-easy-pos", type=int, default=60)
    ap.add_argument("--n-easy-neg", type=int, default=60)
    ap.add_argument("--n-band", type=int, default=80,
                    help="hard cases from the escalation band -- oversample "
                         "these; they are where alpha is earned")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out or config.MATCH_PAIRS)
    total_added = total_skipped = 0
    for db in args.db:
        if not Path(db).exists():
            sys.exit(f"no such run DB: {db}")
        session = get_session(db)
        try:
            sources = {s.source_id: s for s in load_sources(session)}
            # accepted_only: alpha is the matcher's error on the pairs the
            # LIVE pass scores, and the live pass only sees gate survivors
            mentions = load_mentions(session, accepted_only=True)
        finally:
            engine = session.get_bind()
            session.close()
            engine.dispose()            # Windows: release the sqlite handle
        rows = sample_pairs(mentions, sources,
                            n_easy_pos=args.n_easy_pos,
                            n_easy_neg=args.n_easy_neg,
                            n_band=args.n_band, seed=args.seed,
                            run_db=str(db))
        added, skipped = append_rows(out, rows)
        total_added += added
        total_skipped += skipped
        print(f"{db}: sampled {len(rows)} "
              f"(+{added} new, {skipped} already queued)")

    everything = load_rows(out)
    unlabeled = sum(1 for r in everything if not (r["label"] or "").strip())
    print(f"\n{out}: {len(everything)} pairs total, {unlabeled} awaiting "
          f"labels -> python scripts/review_er_pairs.py")


if __name__ == "__main__":
    main()
