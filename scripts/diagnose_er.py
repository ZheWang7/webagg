"""Why is entity resolution false-splitting? A zero-LLM, mention-level audit.

For each requested entity's frozen pool, this script rebuilds exactly what
replay() feeds the ER stage (same gate, same fitted matcher), then scores
every blocked candidate pair with the matcher -- no adjudicator calls, no
clustering -- and audits the SHOULD-MERGE pairs: pairs whose surfaces are
near-identical after normalization, which any correct ER must join.

For those pairs it reports, stage by stage:
  * never blocked        -- blocking produced no candidate pair at all
  * theta <= tau_minus   -- the matcher confidently split them
  * band                 -- escalated to the LLM at run time (verdict unknown
                            here; the run's fragile-pair log has it)
  * theta >= tau_plus    -- matcher said merge (so a split here would be
                            clustering's fault)
plus the mean feature vector per bucket and the fitted model's coefficients,
so the feature doing the damage is named by data, not guesswork.

Usage (repo root; offline, seconds per entity):
    python scripts/diagnose_er.py --cohort formd_v2
    python scripts/diagnose_er.py --cohort formd_v2 --entities Expel Illumio --pairs 12
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                   # noqa: E402
from webagg.certify import (_fitted_gate, _fitted_matcher,  # noqa: E402
                            load_runs_index)
from webagg.entity_resolution import (candidate_pairs_logged,  # noqa: E402
                                      features, normalize_name)
from webagg.pipeline import load_mentions, load_sources     # noqa: E402
from webagg.storage import get_session                      # noqa: E402

FEATS = ["name_sim", "part_sim", "same_domain", "emb_cos", "temporal"]


def audit_entity(eid: str, info: dict, lam: dict, n_detail: int) -> None:
    session = get_session(info["db"])
    try:
        gate = _fitted_gate(lam["delta_E"])
        mentions = [m for m in load_mentions(session) if gate.accept(m)]
        sources = {s.source_id: s for s in load_sources(session)}
    finally:
        engine = session.get_bind()
        session.close()
        engine.dispose()

    matcher = _fitted_matcher(lam, {})
    by_id = {m.mention_id: m for m in mentions}
    cand, _log = candidate_pairs_logged(mentions, sources)

    print(f"\n=== {info['entity_name']} ({eid}) -- {len(mentions)} gated "
          f"mentions, {len({s.domain for s in sources.values()})} domains, "
          f"{len(cand)} blocked pairs ===")

    # the fitted model's learned weights, if reachable (settles a lot fast)
    try:
        cc = matcher.clf.calibrated_classifiers_[0]
        est = getattr(cc, "estimator", None) or cc.base_estimator
        coefs = dict(zip(FEATS, est.coef_[0].round(3)))
        print(f"fitted logistic coefficients: {coefs} "
              f"(intercept {est.intercept_[0]:+.3f})")
    except Exception as e:                                  # noqa: BLE001
        print(f"(could not read model coefficients: {e})")

    # SHOULD-MERGE = same normalized surface, different mention, any source.
    # (Conservative: catches 'Expel' vs 'Expel, Inc.' via normalize_name.)
    buckets: dict[str, list[str]] = {}
    for m in mentions:
        buckets.setdefault(normalize_name(m.entity_surface), []).append(
            m.mention_id)
    should = {tuple(sorted(p))
              for ids in buckets.values() if len(ids) > 1
              for p in combinations(ids, 2)}
    if not should:
        print("no same-surface pairs at all -- surfaces themselves diverge; "
              "dumping distinct surfaces instead:")
        for s in sorted({m.entity_surface for m in mentions})[:20]:
            print(f"   {s!r}")
        return

    rows = []                       # (theta, bucket, pair, feature-vector)
    never = 0
    for a, b in sorted(should):
        if (a, b) not in cand:
            never += 1
            rows.append((None, "never-blocked", (a, b), None))
            continue
        x = features(by_id[a], by_id[b], sources)
        th = matcher.score(x)
        bucket = ("merge (>=tau+)" if th >= matcher.tau_plus else
                  "SPLIT (<=tau-)" if th <= matcher.tau_minus else "band")
        rows.append((th, bucket, (a, b), x))

    from collections import Counter
    counts = Counter(r[1] for r in rows)
    print(f"should-merge pairs: {len(rows)}  ->  " +
          ", ".join(f"{k}: {v}" for k, v in counts.most_common()))

    for bucket in ["merge (>=tau+)", "band", "SPLIT (<=tau-)"]:
        vecs = [r[3] for r in rows if r[1] == bucket and r[3] is not None]
        if vecs:
            mean = np.mean(vecs, axis=0).round(3)
            print(f"  mean features [{bucket:>14}]: "
                  + ", ".join(f"{n}={v}" for n, v in zip(FEATS, mean)))

    print(f"\nworst {n_detail} should-merge pairs by theta:")
    scored = sorted((r for r in rows if r[0] is not None), key=lambda r: r[0])
    for th, bucket, (a, b), x in scored[:n_detail]:
        ma, mb = by_id[a], by_id[b]
        da = sources[ma.source_id].domain
        db_ = sources[mb.source_id].domain
        fx = ", ".join(f"{n}={v:.2f}" for n, v in zip(FEATS, x))
        print(f"  theta={th:.3f} [{bucket}] {ma.entity_surface!r}({da}) "
              f"vs {mb.entity_surface!r}({db_})\n"
              f"      {fx}")
    if never:
        print(f"  (+ {never} same-surface pairs never proposed by blocking)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--entities", nargs="*",
                    default=["Expel", "Squarespace", "Illumio"],
                    help="entity ids or name substrings (default: 3 worst "
                         "split offenders from the fidelity diagnosis)")
    ap.add_argument("--pairs", type=int, default=8,
                    help="how many worst pairs to print per entity")
    args = ap.parse_args()

    index = load_runs_index(args.domain or args.cohort)
    lam = config.LTT_GRID[0]
    wanted = []
    for sel in args.entities:
        hits = [eid for eid, info in index.items()
                if sel.lower() in eid.lower()
                or sel.lower() in info["entity_name"].lower()]
        if not hits:
            print(f"!! no pool matches {sel!r} -- skipped")
        wanted.extend(hits)
    for eid in dict.fromkeys(wanted):        # dedupe, keep order
        audit_entity(eid, index[eid], lam, args.pairs)


if __name__ == "__main__":
    main()
