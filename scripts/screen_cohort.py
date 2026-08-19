"""Screen cohort candidates against EDGAR Form D (attempt #3, step 1).

Reads the candidate-names file, resolves each against EDGAR's company
browse (private filers included), gathers Form D stats, and writes the
screening sheet for the HUMAN full-history judgment:

    data/ground_truth/cohort_screen.csv

Resumable: candidates already in the sheet are skipped on re-run, so a
crash or rate-limit mid-way costs nothing (delete a row to re-screen it).
~2 polite requests per candidate; 256 names takes roughly 4-5 minutes.

After screening: open the CSV, fill human_verdict (accept / reject) using
the evidence -- a company famous since 2012 whose first_formd is 2019 has
pre-EDGAR rounds (the Uber failure) and must be rejected; weak_match rows
need the CIK confirmed by eye. Accepted CIKs then feed build_truth.py.

Usage (repo root; needs SEC network access, no LLM):
    python scripts/screen_cohort.py
    python scripts/screen_cohort.py --candidates data/ground_truth/cohort_candidates.txt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                  # noqa: E402
from webagg.cohort_screen import (SHEET_COLUMNS, read_candidates,  # noqa: E402
                                  screen_one, write_sheet)
from webagg.schema_addressable import _default_client      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates",
                    default=str(config.GROUND_TRUTH_DIR
                                / "cohort_candidates.txt"))
    ap.add_argument("--out",
                    default=str(config.GROUND_TRUTH_DIR
                                / "cohort_screen.csv"))
    args = ap.parse_args()

    names = read_candidates(Path(args.candidates))
    out = Path(args.out)
    rows: list[dict] = []
    if out.exists():                     # resume: keep prior rows verbatim
        with out.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    done = {r["candidate"] for r in rows}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} candidates, {len(done)} already screened, "
          f"{len(todo)} to go")

    client = _default_client()           # the polite EDGAR UA
    try:
        for i, name in enumerate(todo, 1):
            row = screen_one(name, client)
            rows.append(row)
            print(f"[{i}/{len(todo)}] {name:<28} -> {row['verdict']:<11}"
                  f"{row['registered_name'][:34]:<36}"
                  f"D-filings={row['n_formd'] or '-'}")
            if i % 10 == 0 or i == len(todo):
                write_sheet(out, rows)   # save every 10: crash-safe
    finally:
        client.close()
        write_sheet(out, rows)

    ok = sum(1 for r in rows if r["verdict"] == "ok")
    weak = sum(1 for r in rows if r["verdict"] == "weak_match")
    print(f"\n{out}: {len(rows)} screened -- {ok} ok, {weak} weak_match, "
          f"{len(rows) - ok - weak} no_match")
    print("next: fill human_verdict in the CSV (full-history judgment), "
          "then feed accepted CIKs to build_truth.py")


if __name__ == "__main__":
    main()
