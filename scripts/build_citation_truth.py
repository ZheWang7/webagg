"""Build and FREEZE the Exp 2-C truth table from OpenAlex.

cohort/papers.txt lines: 'Label | W1234567890' (the OpenAlex work id, from
the work's openalex.org URL). The cutoff date is pre-registered and passed
explicitly, so the frozen list cannot drift as citations accrue.

Usage:
    python scripts/build_citation_truth.py --cohort cite_v1 \
        --papers cohort/papers.txt --cutoff 2026-06-30
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                        # noqa: E402
from webaggs.oracle_openalex import OpenAlexDriver  # noqa: E402
from webaggs.truth import parse_candidate_line, write_truth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--papers", required=True)
    ap.add_argument("--cutoff", required=True,
                    help="YYYY-MM-DD; citing papers published after this "
                         "are OUT of truth (pre-registered)")
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.papers).read_text(
        encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")]
    driver = OpenAlexDriver()
    rows, report = [], []
    for ln in lines:
        label, _ = parse_candidate_line(ln) if "|" in ln else (ln, None)
        wid = ln.split("|", 1)[1].strip() if "|" in ln else None
        if not wid or not wid.upper().startswith("W"):
            raise SystemExit(f"Paper line {ln!r} needs 'Label | W<openalex id>'")
        meta = driver.work_meta(wid)
        citers = driver.citers(wid, args.cutoff)
        print(f"  {meta['title'][:60]!r} ({meta['year']}): "
              f"{len(citers)} citers <= {args.cutoff} "
              f"(lifetime {meta['cited_by_count']})")
        report.append({"paper": meta["title"], "work_id": wid,
                       "year": meta["year"],
                       "citers_at_cutoff": len(citers),
                       "lifetime_cited_by": meta["cited_by_count"]})
        for i, c in enumerate(citers):
            rows.append({"player": meta["title"], "player_id": i,
                         "season": c["key"], "pts": config.CITE_VALUE,
                         "gp": c["year"] or 0})
    csv_path = write_truth(config.DATA_DIR, args.cohort, rows)
    rep = csv_path.parent / "cohort_report.csv"
    with open(rep, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"\nFrozen truth: {csv_path}\nReport: {rep}")


if __name__ == "__main__":
    main()
