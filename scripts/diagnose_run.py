"""Exp 2-C run DIAGNOSIS — read-only, offline, $0. No grading: it never
opens the truth table, so looking at it cannot leak results.

For each finished run of a cohort it answers "why did the stop test end
the way it did?" from the ledger alone:

  1. the stop test's final numbers: U_hat + psi against eps
     (webaggr.pdf §3.3: stop iff U_hat + psi < eps and the frontier is cold)
  2. did both index walks FINISH? pages per index by outcome, aborted
     walks, unavailable indexes
  3. who lists what: records found by both indexes, by one only (these
     are the singletons that keep U_hat up), by prose pages
  4. how many occasions went to index pages vs web searches

Usage:
    python scripts/diagnose_run.py --cohort cite_v1
    python scripts/diagnose_run.py --run data/experiments/exp2c/runs/<id>.sqlite --eps 0.6
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402

EXP_DIR = config.DATA_DIR / "experiments" / "exp2c"


def diagnose(db: Path, eps: float | None) -> None:
    print(f"\n=== {db.name} ===")
    conn = sqlite3.connect(db)
    stop = conn.execute("SELECT step, reason FROM stops").fetchone()
    print(f"stop: {stop}")

    # --- 1. the stop test's own numbers, last step and closest approach ---
    rows = []
    for step, u, extra in conn.execute(
            "SELECT step, value, extra FROM measurements "
            "WHERE metric='U_hat' ORDER BY step"):
        e = json.loads(extra)
        rows.append((step, u, e))
    if not rows:
        print("no U_hat measurements — the run never completed a step")
        conn.close()
        return
    step, u, e = rows[-1]
    n, f1, psi = e.get("N"), e.get("f1"), e.get("psi")
    print(f"[1] last step {step}: N={n}, singletons f1={f1} "
          f"(f1/N={f1 / max(n, 1):.3f}), U_hat={u:.3f}, psi={psi:.3f}, "
          f"U_hat+psi={u + psi:.3f}, frontier hot={e.get('hot')}")
    if eps is not None:
        gap = u + psi - eps
        print(f"    eps={eps}: the stop needed U_hat+psi < {eps}; "
              f"{'MET' if gap < 0 else f'missed by {gap:.3f}'}")
        print(f"    with psi={psi:.3f} the singleton share had to be below "
              f"{eps - psi:.3f}; it was {u:.3f}")
    best = min(rows, key=lambda r: r[1] + r[2]["psi"])
    print(f"    closest approach: step {best[0]}, U_hat+psi="
          f"{best[1] + best[2]['psi']:.3f} (hot={best[2].get('hot')})")

    # --- 2. did the walks finish? ----------------------------------------
    print("[2] index pages by outcome:")
    for dom, st, k in conn.execute(
            "SELECT domain, status, COUNT(*) FROM sources "
            "WHERE source_class='citation_index' GROUP BY 1, 2 ORDER BY 1, 2"):
        print(f"    {dom:26s} {st:8s} x{k}")
    for metric in ("index_unavailable", "index_walk_aborted",
                   "intake_truncated"):
        for stp, val, extra in conn.execute(
                "SELECT step, value, extra FROM measurements WHERE metric=?",
                (metric,)):
            print(f"    {metric} at step {stp}: value={val} {extra}")

    # --- 3. who lists what -------------------------------------------------
    by_key = defaultdict(set)
    for key, dom, cls in conn.execute(
            "SELECT m.season, s.domain, s.source_class FROM mentions m "
            "JOIN sources s ON s.source_id=m.source_id WHERE m.accepted=1"):
        by_key[key].add(dom if cls == "citation_index" else "prose-web")
    combos = Counter(" + ".join(sorted(v)) for v in by_key.values())
    total = sum(combos.values())
    print(f"[3] {total} distinct records, by which sources list them:")
    for combo, k in combos.most_common():
        print(f"    {k:7d} ({k / max(total, 1):5.1%})  {combo}")

    # --- 4. where the occasions went ----------------------------------------
    n_idx = conn.execute("SELECT COUNT(DISTINCT formulation_id) FROM sources "
                         "WHERE source_class='citation_index'").fetchone()[0]
    print(f"[4] occasions: {rows[-1][2].get('T')} issued, {n_idx} of them "
          f"index pages; spent ${rows[-1][2].get('spent')}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--run", default=None, help="path to one .sqlite ledger")
    ap.add_argument("--eps", type=float, default=None)
    args = ap.parse_args()
    if args.run:
        diagnose(Path(args.run), args.eps)
        return
    if not args.cohort:
        raise SystemExit("give --cohort or --run")
    idx_path = EXP_DIR / "runs_index.json"
    if not idx_path.exists():
        raise SystemExit(f"no {idx_path} — nothing has been run")
    idx = json.loads(idx_path.read_text())
    for key, rec in sorted(idx.items()):
        if not rec.get("done") or f"_{args.cohort}_" not in rec["run_id"]:
            continue
        eps = args.eps if args.eps is not None else \
            (rec.get("params") or {}).get("eps")
        print(f"\n{key[:70]} -> {rec['stop_reason']} / {rec['certificate']}")
        diagnose(EXP_DIR / "runs" / f"{rec['run_id']}.sqlite", eps)


if __name__ == "__main__":
    main()
