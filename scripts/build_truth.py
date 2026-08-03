"""Build the withheld-registry answer key (impl guide Sec. 15).

Runs the SAME EDGARDriver the agent would use -- but OUTSIDE the agent,
with as_oracle=True and the deterministic Form D parser from webagg.formd
("one parser, two roles"). Output, per entity (= per stratum g):

    data/ground_truth/<cohort>/truth_cik##########.json
        the answer key: one record per funding round (amendment chains
        already collapsed), the true SUM and COUNT, and full metadata.

    data/ground_truth/<cohort>/manifest.json
        the cohort card: which CIKs, which filters, parse failures (loud),
        and the calibration/validation split of ENTITIES (guide Sec. 15:
        "the same cohort supplies the calibration data for the fidelity
        certificate -- split it into calibration and validation halves").

The oracle's sweep provenance (every fetched filing + every deterministic
mention) additionally lands in data/runs/<cohort>_truth.sqlite via
run_schema_addressable -- the answer key never sits in a DB the agent is
graded on.

The denylist never applies here: the oracle READS the registry; it is the
agent that will be denied it (--deny sec, separate mechanism).

Usage (repo root; .env NOT needed -- no LLM, no Serper; only SEC access):

    python scripts/build_truth.py --cohort formd_v1 --ciks 1181412 1633917
    python scripts/build_truth.py --cohort formd_v1 --tickers ACME BETA --since 2018
    python scripts/build_truth.py --cohort formd_v1 --name-contains "space explor"

SEC etiquette: data.sec.gov rejects anonymous clients. The driver already
sends config.USER_AGENT (a contact-address UA; override via USER_AGENT in
.env if you want your own address on the requests).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.formd import (build_truth_entity, formd_mentions,  # noqa: E402
                          parse_source, save_truth_entity)
from webagg.risk_control import split_cohort               # noqa: E402
from webagg.schema_addressable import (EDGARDriver,        # noqa: E402
                                       run_schema_addressable)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True,
                    help="cohort name -> data/ground_truth/<cohort>/")
    ap.add_argument("--ciks", nargs="*", default=None,
                    help="explicit CIK numbers (fastest: skips the index)")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="exact tickers (Form D filers are mostly private "
                         "and have none -- prefer --ciks)")
    ap.add_argument("--name-contains", default=None,
                    help="case-insensitive substring of the filer name")
    ap.add_argument("--since", default=None,
                    help="only filings on/after this date (YYYY or YYYY-MM-DD)")
    ap.add_argument("--max-keys", type=int, default=None,
                    help="safety cap on filers swept (blocking predicate; "
                         "recorded in the certificate)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the calibration/validation entity split")
    args = ap.parse_args()

    # -- the filter IS the cohort definition; it goes in the manifest verbatim
    query_filter = {
        "ciks": args.ciks,
        "tickers": args.tickers,
        "name_contains": args.name_contains,
        "forms": ["D", "D/A"],          # the whole point of this oracle
        "since": args.since,
    }
    query_filter = {k: v for k, v in query_filter.items() if v}

    # -- side channels filled by the closure below, one fetch doing double
    #    duty: mentions -> truth DB (provenance), parsed filing -> truth table
    filings_by_cik: dict[str, list] = defaultdict(list)
    failures: list[dict] = []           # every unparseable doc, loudly

    def oracle_extract(src, query):
        """extract_fn for the oracle sweep: parse ONCE, feed both outputs."""
        cik = (src.formulation_id or "").removeprefix("edgar:CIK")
        try:
            filings_by_cik[cik].append(parse_source(src))
        except ET.ParseError as e:
            # a broken doc must never silently thin the answer key
            failures.append({"url": src.url, "cik": cik, "error": str(e)})
            return []
        return formd_mentions(src, query)

    print(f"[build_truth] sweeping EDGAR (filter={query_filter}) ...")
    out = run_schema_addressable(
        "form d funding rounds",        # the oracle keeps everything anyway
        EDGARDriver(),
        query_filter=query_filter,
        run_id=args.cohort,
        extract_fn=oracle_extract,
        max_keys=args.max_keys,
        as_oracle=True,
    )

    # -- filings -> per-entity truth tables ---------------------------------
    cohort_dir = config.GROUND_TRUTH_DIR / args.cohort
    entities: dict[str, dict] = {}
    for cik in sorted(filings_by_cik):
        entity_id = f"cik{cik}"
        truth, meta = build_truth_entity(entity_id, filings_by_cik[cik])
        save_truth_entity(cohort_dir, meta)
        entities[entity_id] = meta

    # -- the entity-level calibration/validation split (Sec. 13 <- Sec. 15).
    #    split_cohort is the SAME function learn_then_test's harness uses,
    #    so the split rule is defined in exactly one place.
    ids = sorted(entities)              # deterministic input order + seeded
    cal, val = split_cohort(ids, seed=args.seed)   # permutation = reproducible

    manifest = {
        "cohort": args.cohort,
        "built_at": datetime.utcnow().isoformat(),
        "query_filter": query_filter,
        "seed": args.seed,
        "split": {"calibration": sorted(cal), "validation": sorted(val)},
        "keys_swept": out["keys_swept"],
        "blocked": out["blocked"],                  # True -> certificate
        "blocking_predicate": out["blocking_predicate"],  # covers closure(K')
        "truth_db": out["db_path"],
        "parse_failures": failures,
        "entities": {eid: {"entity_name": m["entity_name"],
                           "n_rounds": m["n_rounds"],
                           "true_sum": m["true_sum"]}
                     for eid, m in entities.items()},
    }
    (cohort_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))

    # -- the human-readable receipt -----------------------------------------
    print(f"\n{'entity':<16}{'name':<34}{'rounds':>7}{'true SUM':>16}")
    for eid, m in entities.items():
        print(f"{eid:<16}{m['entity_name'][:32]:<34}"
              f"{m['n_rounds']:>7}{m['true_sum']:>16,.0f}")
    print(f"\ncohort dir : {cohort_dir}")
    print(f"truth db   : {out['db_path']}")
    print(f"split      : {len(cal)} calibration / {len(val)} validation "
          f"(seed={args.seed})")
    if failures:
        print(f"\n!! {len(failures)} PARSE FAILURES -- the answer key is "
              f"INCOMPLETE until these are resolved:")
        for f in failures:
            print(f"   {f['cik']}  {f['url']}  ({f['error']})")
        sys.exit(1)                     # incomplete key = failed build, loudly


if __name__ == "__main__":
    main()
