"""Exp 2-C pilot DIAGNOSIS: why was the discovery rate what it was?

Reads the pilot ledger (no network, no spend) and prints the four things
that explain a low rate, in order of likelihood:

  1. fetch outcomes       -- were the dense citation pages DENIED/blocked?
                             (the paywall instrument, webaggr.pdf §7.2: a
                             censored fetch is censored, not absence)
  2. yielding domains     -- which pages actually produced accepted records?
  3. rejection pressure   -- did pages come back but the reader refused rows?
  4. the queries tried    -- did the seed/follow-up formulations even aim at
                             dense sources (citation indexes, reference lists)?

Usage:
    python scripts/pilot_diagnose.py            # newest pilot ledger
    python scripts/pilot_diagnose.py --run data/pilot_exp2c/runs/pilot_x.sqlite
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402

RUNS = config.DATA_DIR / "pilot_exp2c" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="path to a pilot .sqlite ledger")
    args = ap.parse_args()

    # pick the newest pilot ledger unless one was named
    if args.run:
        db = Path(args.run)
    else:
        cands = sorted(RUNS.glob("pilot_*.sqlite"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit(f"no pilot ledgers under {RUNS}")
        db = cands[-1]
    print(f"ledger: {db}\n")
    conn = sqlite3.connect(db)

    # --- run shape: how far it got and what ended it ----------------------
    rid = conn.execute("SELECT run_id FROM measurements LIMIT 1").fetchone()
    rid = rid[0] if rid else "?"
    stop = conn.execute("SELECT step, reason, extra FROM stops").fetchone()
    steps = conn.execute("SELECT MAX(step) FROM measurements").fetchone()[0]
    spent = 0.0
    for (extra,) in conn.execute(
            "SELECT extra FROM measurements WHERE metric='U_hat'"):
        try:
            spent = max(spent, json.loads(extra).get("spent") or 0.0)
        except (json.JSONDecodeError, TypeError):
            pass
    print(f"run {rid}: last step {steps}, stop={stop}, spent ~${spent:.2f}")

    # --- 1. fetch outcomes: the censoring instrument ----------------------
    print("\n[1] fetch outcomes (status x domain) — 'denied'/'paywalled' on "
          "citation-index domains is the smoking gun:")
    tot = Counter(s for (s,) in conn.execute("SELECT status FROM fetch_outcomes"))
    print(f"    totals: {dict(tot)}")
    for st, dom, n in conn.execute(
            "SELECT status, domain, COUNT(*) FROM fetch_outcomes "
            "GROUP BY status, domain ORDER BY COUNT(*) DESC LIMIT 20"):
        print(f"    {st:10s} {n:3d}  {dom}")

    # --- 2. where the accepted records actually came from -----------------
    print("\n[2] accepted-record yield by source domain:")
    rows = conn.execute(
        "SELECT s.domain, COUNT(*) FROM mentions m "
        "JOIN sources s ON s.source_id=m.source_id "
        "WHERE m.accepted=1 GROUP BY s.domain ORDER BY COUNT(*) DESC").fetchall()
    for dom, n in rows:
        print(f"    {n:3d}  {dom}")
    if not rows:
        print("    (none)")

    # --- 3. rejection pressure: pages read but rows refused ---------------
    nrej = conn.execute(
        "SELECT COUNT(*) FROM mentions WHERE accepted=0").fetchone()[0]
    nacc = conn.execute(
        "SELECT COUNT(*) FROM mentions WHERE accepted=1").fetchone()[0]
    nsrc_ok = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE status='ok'").fetchone()[0]
    print(f"\n[3] reader pressure: {nsrc_ok} pages fetched ok -> "
          f"{nacc} accepted / {nrej} rejected mentions")
    for val, n in conn.execute(
            "SELECT value, COUNT(*) FROM measurements "
            "WHERE metric='rejects' GROUP BY value"):
        print(f"    gate/validator reject events: value={val} x{n}")

    # --- 4. what the agent actually searched for --------------------------
    print("\n[4] formulations the LLM proposed (seed + follow-ups, raw):")
    for purpose, raw in conn.execute(
            "SELECT purpose, raw FROM llm_calls "
            "WHERE purpose IN ('seed','followups') ORDER BY id"):
        txt = (raw or "").strip().replace("\n", " ")
        print(f"  -- {purpose}: {txt[:1200]}")

    # --- novelty per occasion, the Y_CAP raw data --------------------------
    print("\n[novelty] new records per capture occasion:")
    per = Counter(s for (s,) in conn.execute(
        "SELECT step FROM coverage_events"))
    for s in sorted(per):
        print(f"    step {s:3d}: {per[s]}")
    conn.close()


if __name__ == "__main__":
    main()
