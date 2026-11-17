"""Build and FREEZE the Exp 2-C truth table from OpenCitations (the oracle).

cohort/papers.txt lines:  Title | DOI | PMID
(PMID optional but needed for the Europe PMC enumerator.) The cutoff is
pre-registered and must be a 31 December, so the frozen list cannot drift
as citations accrue and is exact at every date precision the oracle uses.

Writes, under data/truth/<cohort>/:
  truth.csv + manifest.json   the frozen answer key (hash-locked)
  targets.json                title/DOI/PMID per paper + the cutoff — what
                              the live run needs to address the indexes
  cohort_report.csv           per paper: raw rows, rows outside the DOI
                              universe / undated / after cutoff, truth size

Usage:
    python scripts/build_citation_truth.py --cohort cite_v1 \
        --papers cohort/papers.txt --cutoff 2024-12-31
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                        # noqa: E402
from webaggs.canon import canonical_doi, normalize_name   # noqa: E402
from webaggs.oracle_opencitations import OpenCitationsDriver  # noqa: E402
from webaggs.truth import write_truth             # noqa: E402


def parse_paper_line(line: str) -> dict:
    """'Title | DOI | PMID' -> {title, doi, pmid}. Loud on a bad line."""
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 2 or not parts[0] or not canonical_doi(parts[1]):
        raise SystemExit(f"Paper line {line!r} needs 'Title | DOI | PMID' "
                         "(PMID optional) with a valid DOI.")
    pmid = parts[2] if len(parts) > 2 and parts[2] else None
    if pmid and not pmid.isdigit():
        raise SystemExit(f"Paper line {line!r}: PMID must be numeric.")
    return {"title": parts[0], "doi": canonical_doi(parts[1]), "pmid": pmid}


def main(driver=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--papers", required=True)
    ap.add_argument("--cutoff", required=True,
                    help="YYYY-12-31; citing works created after this year "
                         "are OUT of truth (pre-registered)")
    args = ap.parse_args()

    lines = [ln.strip() for ln in Path(args.papers).read_text(
        encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")]
    papers = [parse_paper_line(ln) for ln in lines]
    driver = driver or OpenCitationsDriver()
    rows, report, targets = [], [], {}
    for p in papers:
        citers, rep = driver.citers(p["doi"], args.cutoff)
        print(f"  {p['title'][:60]!r}: {len(citers)} DOI-registered citers "
              f"<= {args.cutoff} (raw {rep['raw_rows']}, no DOI "
              f"{rep['no_doi']}, undated {rep['no_date']}, after cutoff "
              f"{rep['after_cutoff']})")
        report.append({"paper": p["title"], "doi": p["doi"],
                       "pmid": p["pmid"] or "", **rep})
        targets[normalize_name(p["title"])] = p
        for i, c in enumerate(citers):
            # truth.csv keeps the shared column names of the core truth
            # module: player = paper, season = record key, pts = 1
            rows.append({"player": p["title"], "player_id": i,
                         "season": c["key"], "pts": config.CITE_VALUE,
                         "gp": c["year"]})
    csv_path = write_truth(config.DATA_DIR, args.cohort, rows)
    (csv_path.parent / "targets.json").write_text(json.dumps(
        {"cutoff": args.cutoff, "oracle": "opencitations",
         "targets": targets}, indent=2), encoding="utf-8")
    rep_path = csv_path.parent / "cohort_report.csv"
    with open(rep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"\nFrozen truth: {csv_path}\nTargets: "
          f"{csv_path.parent / 'targets.json'}\nReport: {rep_path}")


if __name__ == "__main__":
    main()
