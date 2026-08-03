"""Inspect a run's sqlite log: the standard diagnostic panels, read-only.

The run DB is the ground truth of what a run did (the measurement spine,
guide ch. 3). This prints the panels you reach for after every live run:

  * table row counts             -- did each stage produce anything?
  * amount mentions by surface   -- WHO did ER see, with WHAT amounts
                                    (under-merge twins show up here)
  * claims                       -- did any page state an aggregate?
                                    (no claims -> no checksum -> nothing
                                    certifies at small n)
  * sources                      -- domain / class / identity_anchored mix
                                    (no anchored source -> beliefs cap out
                                    at 1-(1-qbar)^k)
  * key measurements             -- stop reason, checksum events, re-key,
                                    extraction agreement

Usage (repo root):

    python scripts/inspect_run.py cli_8257c8c7            # by run id
    python scripts/inspect_run.py data/runs/x.sqlite      # or by path
    python scripts/inspect_run.py cli_8257c8c7 --sql "SELECT * FROM claims"
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _connect(target: str) -> sqlite3.Connection:
    """Resolve a run id or a path to the sqlite file; open READ-ONLY so an
    inspection can never corrupt a log (mode=ro via URI)."""
    p = Path(target)
    if not p.suffix:
        p = Path("data/runs") / f"{target}.sqlite"
    if not p.exists():
        sys.exit(f"no such run DB: {p}")
    return sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)


def _panel(con, title, sql, limit=40):
    print(f"\n=== {title} " + "=" * max(1, 60 - len(title)))
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(limit + 1)
    except sqlite3.Error as e:
        print(f"  (query failed: {e})")
        return
    if not rows:
        print("  (empty)")
        return
    print("  " + " | ".join(cols))
    for r in rows[:limit]:
        print("  " + " | ".join("" if v is None else str(v) for v in r))
    if len(rows) > limit:
        print(f"  ... (showing first {limit})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="run id (cli_xxxxxxxx) or path to .sqlite")
    ap.add_argument("--sql", default=None,
                    help="run ONE arbitrary read-only query instead of "
                         "the standard panels")
    args = ap.parse_args()
    con = _connect(args.run)

    if args.sql:
        _panel(con, "custom query", args.sql, limit=200)
        return

    # ---- row counts per table: the run's shape at a glance --------------
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print("=== tables " + "=" * 50)
    for t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<22}{n:>7} rows")

    # ---- the standard panels (column names per webagg/storage.py) -------
    _panel(con, "amount mentions by entity surface",
           "SELECT entity_surface, record_kind, value, currency, "
           "COUNT(DISTINCT source_id) AS n_sources "
           "FROM mentions WHERE attribute='amount' "
           "GROUP BY entity_surface, record_kind, value "
           "ORDER BY entity_surface, value")
    _panel(con, "claims (aggregate statements -> checksum fuel)",
           "SELECT stratum_surface, functional, value_num, scope, "
           "tolerance, source_id FROM claims")
    _panel(con, "sources (class / anchor mix)",
           "SELECT domain, source_class, identity_anchored, doc_type "
           "FROM sources ORDER BY identity_anchored DESC, domain")
    _panel(con, "key measurements",
           "SELECT step, metric, value, stratum, extra FROM measurements "
           "WHERE metric IN ('stop','checksum_certified',"
           "'checksum_certified_post_er','checksum_revoked',"
           "'count_sensitivity_veto','strata_rekey','extract_agreed',"
           "'answer_two_term','q_refined_sources') "
           "ORDER BY step")


if __name__ == "__main__":
    main()
