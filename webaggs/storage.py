"""Per-run SQLite ledger (impl guide §4.2), stdlib sqlite3 — no ORM.

Every live run writes everything needed to replay the stopping decision
offline: per-step U_hat measurements with their full context, coverage
events, claims, stops, fetch outcomes and raw LLM replies. Replays and
plots read this DB and never refetch (the "frozen ledger" discipline).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, url TEXT, domain TEXT, fetch_time TEXT,
    formulation_id TEXT, status TEXT, source_class TEXT, main_text TEXT);
CREATE TABLE IF NOT EXISTS rejected_sources (          -- phi-audit raw material (§7.1)
    source_id TEXT PRIMARY KEY, url TEXT, rejection_score REAL, main_text TEXT);
CREATE TABLE IF NOT EXISTS mentions (
    mention_id TEXT PRIMARY KEY, source_id TEXT, entity_surface TEXT,
    season TEXT, attribute TEXT, value INTEGER, confidence REAL, accepted INTEGER);
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY, source_id TEXT, domain TEXT,
    stratum TEXT, kind TEXT, value REAL);
CREATE TABLE IF NOT EXISTS measurements (              -- the measurement spine
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, step INTEGER, stratum TEXT, metric TEXT, value REAL, extra TEXT);
CREATE TABLE IF NOT EXISTS coverage_events (           -- record first seen at step t
    run_id TEXT, step INTEGER, record_key TEXT, formulation_id TEXT);
CREATE TABLE IF NOT EXISTS stops (
    run_id TEXT PRIMARY KEY, step INTEGER, reason TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS fetch_outcomes (            -- paywall/denial instrument
    run_id TEXT, step INTEGER, url TEXT, domain TEXT, status TEXT, content_len INTEGER);
CREATE TABLE IF NOT EXISTS llm_calls (                 -- raw replies ALWAYS persisted
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, purpose TEXT, raw TEXT, model TEXT);
CREATE TABLE IF NOT EXISTS resolved_records (
    run_id TEXT, record_key TEXT, season TEXT, value INTEGER,
    belief REAL, n_domains INTEGER, PRIMARY KEY (run_id, record_key));
"""


def fresh_ledger(runs_dir: Path, run_id: str) -> "Ledger":
    """Crash-resume hygiene: a ledger file left by a run that never finished
    (crash mid-spend) must NOT be appended to — a second trajectory under the
    same run_id would interleave steps and corrupt every offline replay.
    The stale file is moved aside loudly (<run_id>.crashed-N.sqlite, kept for
    diagnosis — raw evidence is never deleted) and a clean ledger is opened."""
    runs_dir = Path(runs_dir)
    path = runs_dir / f"{run_id}.sqlite"
    if path.exists():
        n = 1
        while (runs_dir / f"{run_id}.crashed-{n}.sqlite").exists():
            n += 1
        crashed = runs_dir / f"{run_id}.crashed-{n}.sqlite"
        path.rename(crashed)
        print(f"[warn] unfinished ledger for {run_id} moved to {crashed.name}; "
              "starting a clean run (the partial spend is a loss, the record "
              "stays clean).")
    return Ledger(path)


class Ledger:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # --- writers ---------------------------------------------------------
    def add_source(self, s) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?)",
            (s.source_id, s.url, s.domain, s.fetch_time, s.formulation_id,
             s.status, s.source_class, s.main_text))

    def add_rejected(self, source_id, url, score, text) -> None:
        self.conn.execute("INSERT OR REPLACE INTO rejected_sources VALUES (?,?,?,?)",
                          (source_id, url, score, text))

    def add_mention(self, m, accepted: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO mentions VALUES (?,?,?,?,?,?,?,?)",
            (m.mention_id, m.source_id, m.entity_surface, m.season,
             m.attribute, m.value, m.confidence, int(accepted)))

    def add_claim(self, c) -> None:
        self.conn.execute("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?)",
                          (c.claim_id, c.source_id, c.domain, c.stratum, c.kind, c.value))

    def log_measurement(self, run_id, step, stratum, metric, value, extra=None) -> None:
        self.conn.execute(
            "INSERT INTO measurements (run_id, step, stratum, metric, value, extra) "
            "VALUES (?,?,?,?,?,?)",
            (run_id, step, stratum, metric, float(value), json.dumps(extra or {})))

    def log_coverage(self, run_id, step, record_key, formulation_id) -> None:
        self.conn.execute("INSERT INTO coverage_events VALUES (?,?,?,?)",
                          (run_id, step, record_key, formulation_id))

    def log_stop(self, run_id, step, reason, extra=None) -> None:
        self.conn.execute("INSERT OR REPLACE INTO stops VALUES (?,?,?,?)",
                          (run_id, step, reason, json.dumps(extra or {})))

    def log_fetch_outcome(self, run_id, step, url, domain, status, content_len) -> None:
        self.conn.execute("INSERT INTO fetch_outcomes VALUES (?,?,?,?,?,?)",
                          (run_id, step, url, domain, status, content_len))

    def log_llm(self, run_id, purpose, raw, model) -> None:
        self.conn.execute(
            "INSERT INTO llm_calls (run_id, purpose, raw, model) VALUES (?,?,?,?)",
            (run_id, purpose, raw, model))

    def add_resolved(self, run_id, record_key, season, value, belief, n_domains) -> None:
        self.conn.execute("INSERT OR REPLACE INTO resolved_records VALUES (?,?,?,?,?,?)",
                          (run_id, record_key, season, value, belief, n_domains))

    # --- readers (replay path) --------------------------------------------
    def measurements(self, run_id, metric=None):
        q = "SELECT step, stratum, metric, value, extra FROM measurements WHERE run_id=?"
        args = [run_id]
        if metric:
            q += " AND metric=?"
            args.append(metric)
        q += " ORDER BY step"
        return [(s, g, m, v, json.loads(e)) for s, g, m, v, e in
                self.conn.execute(q, args).fetchall()]

    def coverage(self, run_id):
        return self.conn.execute(
            "SELECT step, record_key FROM coverage_events WHERE run_id=? ORDER BY step",
            (run_id,)).fetchall()

    def stop(self, run_id):
        row = self.conn.execute(
            "SELECT step, reason, extra FROM stops WHERE run_id=?", (run_id,)).fetchone()
        return None if row is None else (row[0], row[1], json.loads(row[2]))

    def resolved(self, run_id):
        return self.conn.execute(
            "SELECT record_key, season, value, belief, n_domains "
            "FROM resolved_records WHERE run_id=?", (run_id,)).fetchall()

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        # Windows: SQLite holds a file lock until the connection is disposed;
        # close before any unlink (methodology notes).
        self.conn.commit()
        self.conn.close()
