"""Price the band before paying for it: count adjudications a recert needs.

After the matcher refit (2026-09-07) most same-entity pairs land in the
escalation band, so replay cost is dominated by LLM adjudications. This
script computes, entirely OFFLINE (no LLM):

  * per grid config: how many blocked pairs fall in its band (tau-, tau+)
  * WITHOUT a cache: the sum over configs (each config re-asks)
  * WITH the pair-level cache: unique pairs in any config's band (each
    pair asked once across the whole certification)

and turns both into dollar and wall-clock estimates.

Usage (repo root):
    python scripts/estimate_adjudication_load.py --cohort formd_v2
    python scripts/estimate_adjudication_load.py --cohort formd_v2 --cost-per-call 0.005 --secs-per-call 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                   # noqa: E402
from webagg.certify import (_fitted_gate, _fitted_matcher,  # noqa: E402
                            load_runs_index)
from webagg.entity_resolution import (candidate_pairs,      # noqa: E402
                                      features)
from webagg.pipeline import load_mentions, load_sources     # noqa: E402
from webagg.storage import get_session                      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--cost-per-call", type=float, default=0.003,
                    help="USD per adjudication call (default 0.003)")
    ap.add_argument("--secs-per-call", type=float, default=1.5,
                    help="expected seconds per adjudication call")
    args = ap.parse_args()

    index = load_runs_index(args.domain or args.cohort)
    grid = config.LTT_GRID
    matcher = _fitted_matcher(grid[0], {})   # theta doesn't depend on tau

    per_lam = [0] * len(grid)
    unique_any = 0
    total_pairs = 0
    print(f"{'entity':<30}{'pairs':>7}" +
          "".join(f"  band@l{i}" for i in range(len(grid))) + "  uniq")
    for eid, info in sorted(index.items()):
        session = get_session(info["db"])
        try:
            gate = _fitted_gate(grid[0]["delta_E"])
            mentions = [m for m in load_mentions(session) if gate.accept(m)]
            sources = {s.source_id: s for s in load_sources(session)}
        finally:
            engine = session.get_bind()
            session.close()
            engine.dispose()
        if not mentions:
            continue
        by_id = {m.mention_id: m for m in mentions}
        thetas = []
        for a, b in sorted(candidate_pairs(mentions, sources)):
            thetas.append(matcher.score(features(by_id[a], by_id[b], sources)))
        counts = []
        uniq = 0
        for th in thetas:
            in_band = [lam["tau_minus"] < th < lam["tau_plus"] for lam in grid]
            for i, hit in enumerate(in_band):
                per_lam[i] += hit
            uniq += any(in_band)
        unique_any += uniq
        total_pairs += len(thetas)
        counts = [sum(lam["tau_minus"] < th < lam["tau_plus"]
                      for th in thetas) for lam in grid]
        print(f"{info['entity_name'][:29]:<30}{len(thetas):>7}" +
              "".join(f"{c:>9}" for c in counts) + f"{uniq:>6}")

    no_cache = sum(per_lam)
    fmt = lambda n: (f"${n * args.cost_per_call:,.0f}, "        # noqa: E731
                     f"~{n * args.secs_per_call / 3600:.1f} h")
    print(f"\nblocked pairs total     : {total_pairs}")
    for i, lam in enumerate(grid):
        print(f"band at config {i} (tau {lam['tau_minus']}/{lam['tau_plus']})"
              f" : {per_lam[i]}")
    print(f"adjudications WITHOUT cache: {no_cache:>7}  ({fmt(no_cache)})")
    print(f"adjudications WITH cache   : {unique_any:>7}  ({fmt(unique_any)})")
    print("(rates are your flags; defaults are rough. Sequential wall-clock; "
          "the run interleaves other work, so treat hours as an upper bound.)")


if __name__ == "__main__":
    main()
