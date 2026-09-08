"""Harvest deployment-representative ER pairs from the frozen A3 pools.

WHY (er_diagnosis 2026-09-07): the A2 labeled set was drawn from same-page
aggregator runs, so name features had ~zero variance across it and the
fitted matcher learned to gate merges on same_domain + temporal. In
deployment, the pairs ER exists to merge are the OPPOSITE shape: one
company across different sites and different years. This script mines the
34 frozen calibration pools for exactly the missing strata and queues them
for human labeling in the existing match_pairs.csv workflow.

Three strata (bucket names record provenance; the HUMAN verdict is the
only label, as always):

  xdom_pos  same entity's pool, same normalized surface, different
            domains -- the cross-site pairs deployment is made of.
            Biased toward temporally DISTANT pairs, so the temporal
            feature finally sees same-company variance.
  xent_neg  mentions from two different entities' pools -- known-different
            companies. Biased toward high name similarity (Magic AI vs
            Magic Spoon; the *Health cluster): the hard negatives that
            teach the model names are necessary, not sufficient.
  band      same-entity pairs the CURRENT fitted matcher scores inside
            (tau-, tau+): today's fragile decisions, worth pinning.

Appends to data/ground_truth/match_pairs.csv via append_rows (deduped on
pair_id; existing human labels always win). Then label with:

    python scripts/review_er_pairs.py

Offline: no LLM, no network beyond the local embedding model. NOTE: the
theta_cold column is filled with the CURRENT (fitted, alpha=0.1321)
matcher's score at harvest time, not the hand-tuned cold score -- it
records what today's model believed, which is the interesting baseline.

Usage (repo root):
    python scripts/harvest_er_pairs.py --cohort formd_v2
    python scripts/harvest_er_pairs.py --cohort formd_v2 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidfuzz.fuzz import token_set_ratio                  # noqa: E402

from webagg import config                                   # noqa: E402
from webagg.certify import (_fitted_gate, _fitted_matcher,  # noqa: E402
                            load_runs_index)
from webagg.entity_resolution import (candidate_pairs,      # noqa: E402
                                      features, normalize_name)
from webagg.er_pairs import (FEATURE_NAMES, _pair_id,       # noqa: E402
                             append_rows)
from webagg.pipeline import load_mentions, load_sources     # noqa: E402
from webagg.storage import get_session                      # noqa: E402

MAX_PER_ENTITY_PAIR = 6      # don't let one look-alike duo flood the queue
MAX_PER_ENTITY = 10          # ... nor one entity dominate a stratum


def _load_pool(info: dict, delta_e: float):
    """One entity's gated mentions + sources, exactly as replay sees them."""
    session = get_session(info["db"])
    try:
        gate = _fitted_gate(delta_e)
        mentions = [m for m in load_mentions(session) if gate.accept(m)]
        sources = {s.source_id: s for s in load_sources(session)}
    finally:
        engine = session.get_bind()
        session.close()
        engine.dispose()
    return mentions, sources


def _row(m_a, m_b, lookup, matcher, bucket, run_db):
    x = features(m_a, m_b, lookup)
    a, b = sorted((m_a.mention_id, m_b.mention_id))
    if a != m_a.mention_id:
        m_a, m_b = m_b, m_a
    return {
        "pair_id": _pair_id(a, b),
        "mention_a": a, "mention_b": b,
        "surface_a": m_a.entity_surface, "surface_b": m_b.entity_surface,
        "kind_a": m_a.record_kind, "kind_b": m_b.record_kind,
        "domain_a": lookup[m_a.source_id].domain,
        "domain_b": lookup[m_b.source_id].domain,
        "passage_a": (m_a.passage or "")[:240],
        "passage_b": (m_b.passage or "")[:240],
        **{name: f"{val:.6f}" for name, val in zip(FEATURE_NAMES, x)},
        "theta_cold": f"{matcher.score(x):.6f}",
        "bucket": bucket,
        "run_db": run_db,
        "label": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--n-xdom-pos", type=int, default=70)
    ap.add_argument("--n-xent-neg", type=int, default=70)
    ap.add_argument("--n-band", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue composition; write nothing")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    lam = config.LTT_GRID[0]
    matcher = _fitted_matcher(lam, {})
    index = load_runs_index(args.domain or args.cohort)

    pools = {}                                    # eid -> (mentions, sources)
    for eid, info in sorted(index.items()):
        mentions, sources = _load_pool(info, lam["delta_E"])
        if mentions:
            pools[eid] = (mentions, sources, info)
    print(f"[harvest] {len(pools)} pools with gated mentions "
          f"(of {len(index)} in the runs index)")

    rows: list[dict] = []

    # ---- stratum 1: xdom_pos -- cross-domain, same surface, same entity ----
    per_entity: dict[str, list] = defaultdict(list)
    for eid, (mentions, sources, info) in pools.items():
        by_surface: dict[str, list] = defaultdict(list)
        for m in mentions:
            by_surface[normalize_name(m.entity_surface)].append(m)
        cands = []
        for ms in by_surface.values():
            for m_a, m_b in combinations(ms, 2):
                if sources[m_a.source_id].domain != sources[m_b.source_id].domain:
                    cands.append((m_a, m_b))
        if not cands:
            continue
        # bias toward temporally distant pairs: compute features once,
        # keep the lowest-temporal half plus a random sprinkle of the rest
        scored = []
        for m_a, m_b in cands:
            x = features(m_a, m_b, sources)
            scored.append((x[FEATURE_NAMES.index("f_temporal")]
                           if "f_temporal" in FEATURE_NAMES else x[-1],
                           m_a, m_b))
        scored.sort(key=lambda t: t[0])
        keep = scored[:MAX_PER_ENTITY]
        per_entity[eid] = [(m_a, m_b, sources, info["db"]) for _, m_a, m_b in keep]

    # round-robin across entities until the quota fills
    quota, exhausted = args.n_xdom_pos, False
    while quota > 0 and not exhausted:
        exhausted = True
        for eid in sorted(per_entity):
            if per_entity[eid] and quota > 0:
                m_a, m_b, lookup, db = per_entity[eid].pop(0)
                rows.append(_row(m_a, m_b, lookup, matcher, "xdom_pos", db))
                quota -= 1
                exhausted = False

    # ---- stratum 2: xent_neg -- cross-entity, name-similarity-ranked ------
    # rank at the SURFACE level first (cheap), then materialize mention pairs
    surf = []                                     # (eid, surface, one mention)
    for eid, (mentions, sources, info) in pools.items():
        for s in {m.entity_surface for m in mentions}:
            surf.append((eid, s))
    pair_scores = []
    for (ea, sa), (eb, sb) in combinations(surf, 2):
        if ea == eb:
            continue
        pair_scores.append((token_set_ratio(sa, sb), ea, sa, eb, sb))
    pair_scores.sort(reverse=True)
    n_hard = int(args.n_xent_neg * 0.7)
    picks = pair_scores[:n_hard * 3]              # oversample, then cap below
    rng.shuffle(pair_scores)
    picks += pair_scores[:args.n_xent_neg]        # the random arm

    taken_pair: dict[tuple, int] = defaultdict(int)
    taken_ent: dict[str, int] = defaultdict(int)
    n_neg = 0
    for _, ea, sa, eb, sb in picks:
        if n_neg >= args.n_xent_neg:
            break
        key = tuple(sorted((ea, eb)))
        if taken_pair[key] >= MAX_PER_ENTITY_PAIR:
            continue
        if taken_ent[ea] >= 3 * MAX_PER_ENTITY or taken_ent[eb] >= 3 * MAX_PER_ENTITY:
            continue
        ma_all, sa_lookup, ia = pools[ea]
        mb_all, sb_lookup, ib = pools[eb]
        m_a = next(m for m in ma_all if m.entity_surface == sa)
        m_b = next(m for m in mb_all if m.entity_surface == sb)
        lookup = {**sa_lookup, **sb_lookup}
        rows.append(_row(m_a, m_b, lookup, matcher, "xent_neg",
                         f"{ia['db']}|{ib['db']}"))
        taken_pair[key] += 1
        taken_ent[ea] += 1
        taken_ent[eb] += 1
        n_neg += 1

    # ---- stratum 3: band -- today's fragile same-entity decisions ---------
    taken_ids = {r["pair_id"] for r in rows}      # earlier strata win
    band_cands = []
    for eid, (mentions, sources, info) in pools.items():
        by_id = {m.mention_id: m for m in mentions}
        for a, b in sorted(candidate_pairs(mentions, sources)):
            if _pair_id(*sorted((a, b))) in taken_ids:
                continue
            x = features(by_id[a], by_id[b], sources)
            th = matcher.score(x)
            if matcher.tau_minus < th < matcher.tau_plus:
                band_cands.append((by_id[a], by_id[b], sources, info["db"]))
    if band_cands:
        idx = rng.choice(len(band_cands),
                         size=min(args.n_band, len(band_cands)),
                         replace=False)
        for i in sorted(idx):
            m_a, m_b, lookup, db = band_cands[i]
            rows.append(_row(m_a, m_b, lookup, matcher, "band", db))

    # ---- dedupe within the harvest, report, write -------------------------
    uniq: dict[str, dict] = {}
    for r in rows:                        # first stratum to claim a pair wins
        uniq.setdefault(r["pair_id"], r)
    rows = list(uniq.values())
    from collections import Counter
    comp = Counter(r["bucket"] for r in rows)
    print(f"[harvest] queue composition: {dict(comp)}  ({len(rows)} unique)")

    if args.dry_run:
        for r in rows[:12]:
            print(f"  [{r['bucket']:>8}] {r['surface_a']!r}({r['domain_a']}) "
                  f"vs {r['surface_b']!r}({r['domain_b']})  "
                  f"theta_now={r['theta_cold']}")
        print("(dry run -- nothing written)")
        return

    added, skipped = append_rows(config.MATCH_PAIRS, rows)
    print(f"[harvest] match_pairs.csv: +{added} new, {skipped} already "
          f"present (existing labels untouched)")
    print("next: python scripts/review_er_pairs.py")


if __name__ == "__main__":
    main()
