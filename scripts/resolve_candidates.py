"""Resolve and VERIFY cohort candidate ids against the NHL's own systems.

Dev-time helper (never an experiment path): candidates.txt lines may be
'Name' (no id yet) or 'Name | ID'. For each line this script

  1. resolves a bare Name -> id via the league's player-search endpoint,
     accepting it only when EXACTLY ONE result matches the name (nickname
     variants folded by the same deterministic rule ER uses);
  2. verifies EVERY id (given or resolved) via the landing endpoint: the
     API's official display name must match the label, and the career rows
     come back self-validated against careerTotals (NHLDriver);
  3. rewrites candidates.txt with resolved ids and writes
     cohort/resolution_report.csv (name, id, seasons, GP, points, and the
     PREREGISTRATION §4 bucket: star >= 500 GP / fringe <= 150 GP / neither).

Exit nonzero if anything is unresolved, ambiguous, or mismatched — the
report still lists everything, so one run tells you exactly what to fix.
Runs from CI (open egress) or any machine that can reach api-web.nhle.com.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs.canon import normalize_name              # noqa: E402
from webaggs.er import resolve_entity                 # noqa: E402
from webaggs.oracle_nhl import NHLDriver, OracleError  # noqa: E402
from webaggs.truth import parse_candidate_line        # noqa: E402

SEARCH_URL = "https://search.d3.nhle.com/api/v1/search/player"
CANDIDATES = Path(__file__).resolve().parents[1] / "cohort" / "candidates.txt"
REPORT = CANDIDATES.parent / "resolution_report.csv"


def names_match(a: str, b: str) -> bool:
    """Exact normalized match, or the deterministic ER fold in either
    direction (handles Josh/Joshua-style official-name variants)."""
    if normalize_name(a) == normalize_name(b):
        return True
    return (resolve_entity(a, b) is not None
            or resolve_entity(b, a) is not None)


def _result_fields(item: dict):
    """Defensive parse of one search result: the id and name keys have
    varied across API versions. Returns (id, name) or None — and the
    caller prints the raw item when nothing parses (loud, never a guess)."""
    pid = item.get("playerId") or item.get("playerID") or item.get("id")
    name = item.get("name")
    if name is None and ("firstName" in item or "lastName" in item):
        fn, ln = item.get("firstName"), item.get("lastName")
        fn = fn.get("default") if isinstance(fn, dict) else fn
        ln = ln.get("default") if isinstance(ln, dict) else ln
        name = f"{fn or ''} {ln or ''}".strip()
    try:
        return (int(str(pid)), str(name)) if pid and name else None
    except ValueError:
        return None


def search_player(name: str) -> list[tuple[int, str]]:
    resp = requests.get(SEARCH_URL,
                        params={"culture": "en-us", "limit": 20, "q": name},
                        timeout=20)
    resp.raise_for_status()
    items = resp.json()
    out, unparsed = [], []
    for item in items if isinstance(items, list) else []:
        parsed = _result_fields(item)
        (out if parsed else unparsed).append(parsed or item)
    if not out and unparsed:
        print(f"  [schema?] could not parse search results for {name!r}; "
              f"first raw item: {unparsed[0]!r}")
    return out


def main() -> int:
    lines = [ln.strip() for ln in CANDIDATES.read_text(encoding="utf-8")
             .splitlines()]
    driver = NHLDriver(pace_seconds=1.0)
    rows, failures, out_lines = [], [], []

    for ln in lines:
        if not ln or ln.startswith("#"):
            out_lines.append(ln)
            continue
        label, pid = parse_candidate_line(ln)

        if pid is None:                               # factor 1: name -> id
            time.sleep(0.5)
            results = search_player(label)
            exact = sorted({i for i, n in results
                            if normalize_name(n) == normalize_name(label)})
            matches = [(i, n) for i, n in results if names_match(label, n)]
            ids = exact if len(exact) == 1 else sorted({i for i, _ in matches})
            if len(ids) != 1:
                failures.append(f"{label}: {'AMBIGUOUS ' + str(matches[:6]) if ids else 'NOT FOUND in league search'}")
                out_lines.append(ln)                  # leave line untouched
                continue
            pid = ids[0]

        try:                                          # factor 2: id -> name
            official, seasons = driver.career_regular_season(pid)
        except OracleError as e:
            failures.append(f"{label} | {pid}: {e}")
            out_lines.append(ln)
            continue
        if not names_match(label, official):
            failures.append(f"{label} | {pid}: API says this id is "
                            f"{official!r} — WRONG PLAYER, fix the line")
            out_lines.append(ln)
            continue

        gp = sum(r["gp"] for r in seasons)
        pts = sum(r["pts"] for r in seasons)
        last = max(r["season"] for r in seasons)
        bucket = ("star" if gp >= 500 else
                  "fringe" if gp <= 150 else "NEITHER (prereg §4)")
        rows.append({"player": official, "player_id": pid,
                     "n_seasons": len(seasons), "career_gp": gp,
                     "career_pts": pts, "last_season": last,
                     "bucket": bucket})
        out_lines.append(f"{official} | {pid}")
        print(f"  OK  {official:<24} {pid}  {len(seasons):2d} seasons "
              f"{gp:4d} GP {pts:4d} pts  last {last}  [{bucket}]")

    CANDIDATES.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    if rows:
        with open(REPORT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} verified -> {REPORT.name}; "
              f"candidates.txt rewritten with ids.")
    if failures:
        print(f"\n{len(failures)} FAILURES (lines left untouched):")
        for f_ in failures:
            print(f"  !! {f_}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
