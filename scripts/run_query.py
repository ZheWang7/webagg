"""
Runs the full end-to-end pipeline on one query and prints, per stratum:
its total, its own +/- halfwidth, and the regime label that says WHY the
interval is what it is (registry / checksum COUNT / checksum SUM /
statistical) -- plus the forgery margin kappa as a per-cell diagnostic.
Strata that earned no certificate print ABANDONED with their achieved
U_hat + psi. Then the global line, the two eps terms with the eps_F
provenance tag, and the top-B human checks from the verification
allocator.

Usage (repo root, .env with the API keys):

    python scripts/run_query.py "funding rounds of Acme Robotics" \
        --attrs amount,date,stage --domain startup_funding --max-steps 40

    python scripts/run_query.py "..." --surface-er     # no torch installed:
                                                       # group by normalized
                                                       # entity surface

The guide's own acceptance test for this chapter: "If your CLI prints one
number with one interval, you have not implemented the paper."
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import pipeline, config                       # noqa: E402
from webagg.report import format_report                   # noqa: E402


def _surface_cluster(mentions, source_lookup):
    """--surface-er fallback: 'ER' by normalized entity surface only.

    For machines without sentence-transformers/torch. NOTE this is a
    DEGRADED seam (no matcher, no fragile pairs -> the count-sensitivity
    check is inert); fine for a smoke run, not for reported experiments.
    """
    from webagg.frontier import normalize_surface
    return {m.mention_id: normalize_surface(m.entity_surface)
            for m in mentions}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", help="the aggregation query, in plain English")
    ap.add_argument("--run-id", default=None, help="default: fresh uuid")
    ap.add_argument("--mode", default="open_web",
                    choices=["open_web", "schema"])
    ap.add_argument("--attrs", default="amount,date,stage",
                    help="comma-separated query attributes")
    ap.add_argument("--aggregate-attr", default="amount")
    ap.add_argument("--domain", default=None,
                    help="fidelity-certificate domain (Sec. 13); without "
                         "one, eps_F falls back to a labelled constant")
    ap.add_argument("--eps", type=float, default=config.EPS_G)
    ap.add_argument("--delta", type=float, default=config.DELTA_M)
    ap.add_argument("--eta", type=float, default=config.ETA)
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    ap.add_argument("--verify-budget", type=int, default=config.VERIFY_BUDGET)
    ap.add_argument("--surface-er", action="store_true",
                    help="skip the real matcher; cluster by entity surface")
    args = ap.parse_args()

    if args.mode == "schema":
        # A schema sweep needs a concrete driver object (guide ch. 10);
        # the generic CLI has no way to construct one from a string.
        sys.exit("schema mode needs a driver: use a runner script that "
                 "builds one (see tests/sch_addr_sanity.py), or open-web "
                 "mode here.")

    run_id = args.run_id or f"cli_{uuid.uuid4().hex[:8]}"
    result = pipeline.end_to_end(
        args.query, run_id=run_id,
        query_attributes=set(a.strip() for a in args.attrs.split(",")),
        aggregate_attr=args.aggregate_attr, mode=args.mode,
        eps=args.eps, delta=args.delta, eta=args.eta,
        max_steps=args.max_steps, domain=args.domain,
        verify_budget=args.verify_budget,
        cluster_fn=_surface_cluster if args.surface_er else None)

    print(f"\nrun_id: {run_id}  (db: data/runs/{run_id}.sqlite)\n")
    print(format_report(result["report"], result["verify_menu"]))


if __name__ == "__main__":
    main()
