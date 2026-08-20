"""Fill human_verdict on the cohort screening sheet (attempt #3, step 3).

The screen resolved names mechanically; what stays human (deliberately,
see webagg/cohort_screen.py) is TWO judgments per CIK-carrying row:

  1. IDENTITY -- is this registered filer the operating company the
     candidate names? Fund vehicles, casinos, and near-miss spellings
     resolve to the same press name; the alts in the note column list
     the runner-ups the matcher saw.
  2. FULL HISTORY -- does first_formd plausibly cover the company's
     earliest press-known raise? A company famous years before its first
     Form D has pre-EDGAR rounds (the Uber failure) and would poison its
     truth table: reject.

Keys:
  a  accept  -> human_verdict=accept (feeds build_truth.py)
  r  reject  -> human_verdict=reject
  s  skip    -> leave blank, decide later
  b  back    -> revisit the previous row
  q  quit    -> save and show the Hoeffding floor for the accepted count

Every keystroke rewrites the CSV immediately (crash-safe, same philosophy
as the screen itself). no_match rows are never shown -- they carry no CIK
to accept. Re-running resumes at the first unverdicted row.

Usage (repo root, no network, no API keys):

    python scripts/review_screen.py
    python scripts/review_screen.py --all     # include already-verdicted rows
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                   # noqa: E402
from webagg.cohort_screen import SHEET_COLUMNS              # noqa: E402

EDGAR_ROW = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
             "&CIK={cik}&type=D&dateb=&owner=include&count=40")


def _load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _show(i: int, total: int, row: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[{i}/{total}]  {row['candidate']}")
    print(f"  match : {row['registered_name']}  (CIK {row['cik'].lstrip('0')}"
          f", score {row['match_score']}, {row['verdict']})")
    print(f"  Form D: {row['n_formd']} filings ({row['n_original_d']} "
          f"original) spanning {row['first_formd'] or '?'} .. "
          f"{row['last_formd'] or '?'}")
    if row.get("note"):
        print(textwrap.fill("note  : " + row["note"], width=72,
                            subsequent_indent="          "))
    print(f"  edgar : {EDGAR_ROW.format(cik=row['cik'].lstrip('0'))}")
    print("  identity right AND history full?  "
          "[a]ccept [r]eject [s]kip [b]ack [q]uit")


def _hoeffding_floor(n_cal: int) -> float:
    """Smallest eps certifiable at ZERO realized loss with n_cal entities:
    exp(-2 n eps^2) <= DELTA_F  ->  eps >= sqrt(ln(1/DELTA_F) / (2 n))."""
    if n_cal <= 0:
        return float("inf")
    return math.sqrt(math.log(1.0 / config.DELTA_F) / (2.0 * n_cal))


def _summary(rows: list[dict]) -> None:
    acc = sum(1 for r in rows if r["human_verdict"] == "accept")
    rej = sum(1 for r in rows if r["human_verdict"] == "reject")
    dead = sum(1 for r in rows if r["verdict"] != "no_match"
               and r["n_formd"] in ("", "0"))
    todo = sum(1 for r in rows
               if r["verdict"] != "no_match"
               and r["n_formd"] not in ("", "0") and not r["human_verdict"])
    n_cal = acc // 2                      # split_cohort takes halves
    floor = _hoeffding_floor(n_cal)
    print(f"\naccepted={acc}  rejected={rej}  undecided={todo}  "
          f"no_form_d(auto-excluded)={dead}")
    print(f"calibration half at current count: {n_cal} entities")
    print(f"zero-loss Hoeffding floor: eps_F >= {floor:.3f} "
          f"(DELTA_F={config.DELTA_F}); realized loss raises this.")
    print(f"target EPS_F_TARGET={config.EPS_F_TARGET} needs "
          f"{math.ceil(math.log(1/config.DELTA_F)/(2*config.EPS_F_TARGET**2))}"
          " calibration entities at zero loss.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sheet",
                    default=str(config.GROUND_TRUTH_DIR
                                / "cohort_screen.csv"))
    ap.add_argument("--all", action="store_true",
                    help="revisit rows that already carry a verdict")
    args = ap.parse_args()

    path = Path(args.sheet)
    if not path.exists():
        sys.exit(f"no sheet at {path} -- run scripts/screen_cohort.py first")
    rows = _load(path)

    # queue: CIK-carrying rows WITH something to grade, unverdicted unless
    # --all. n_formd=0 rows are mechanically dead (a filer can appear in a
    # type=D browse via DRS/DEF-14A prefix matches yet have zero actual
    # Form Ds -- Cerebras); no human judgment can rescue an empty record.
    queue = [r for r in rows if r["verdict"] != "no_match"
             and r["n_formd"] not in ("", "0")
             and (args.all or not r["human_verdict"])]
    if not queue:
        print("nothing to review")
        _summary(rows)
        return

    i = 0
    while 0 <= i < len(queue):
        _show(i + 1, len(queue), queue[i])
        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            key = "q"
        if key == "a":
            queue[i]["human_verdict"] = "accept"
            _save(path, rows)             # rows share dicts with queue
            i += 1
        elif key == "r":
            queue[i]["human_verdict"] = "reject"
            _save(path, rows)
            i += 1
        elif key == "s":
            i += 1
        elif key == "b":
            i = max(0, i - 1)
        elif key == "q":
            break
        # any other key: re-show the same row
    _save(path, rows)
    _summary(rows)


if __name__ == "__main__":
    main()
