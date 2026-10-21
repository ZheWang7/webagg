"""Core data models (webaggr.pdf §2: sources, mentions, claims).

Plain dataclasses; storage.py mirrors these columns in SQLite.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def sha16(text: str) -> str:
    """First 16 hex of SHA-256 — the identifier convention of the impl guide."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Source:
    """One fetched web page (webaggr.pdf §2: a source covers some records)."""
    source_id: str              # sha16(url|fetch_time)
    url: str
    domain: str
    fetch_time: str             # ISO timestamp
    formulation_id: str         # which search surfaced it (provenance)
    status: str                 # "ok" | "denied" | "paywall" | "empty" | "error"
    main_text: str = ""         # extracted page text ("" unless status == "ok")
    source_class: str = "other" # reliability class (config.CLASS_PRIORS key)


@dataclass
class Mention:
    """One extracted (entity, season, attribute, value) assertion from a source.

    The record key is normalize(entity)|season — a (player, season) record,
    the unit the stopping rule counts (webaggr.pdf §2.4 Definition 4).
    """
    mention_id: str
    source_id: str
    entity_surface: str         # player name as it appears on the page
    season: str                 # canonical "YYYY-YY", e.g. "2018-19"
    attribute: str              # "points" (Exp 2-S uses one attribute)
    value: int                  # integer points (exact-match grading)
    confidence: float           # extractor self-confidence, gated conformally (§7.1)
    extractor_id: str = "llm_v1"

    def record_key(self, entity_norm: str) -> str:
        # record key = entity|season; season is the natural join key the
        # finance domain lacked (webaggr.pdf §6 — here ER is nearly free)
        return f"{entity_norm}|{self.season}"


@dataclass
class Claim:
    """An aggregate claim a page makes about the whole answer (webaggr.pdf §5.2).

    Two kinds in this domain:
      COUNT  — "played N NBA seasons"      -> exact-closure checksum
      SUM    — "career total of X points"  -> missing-value bound Delta+
    """
    claim_id: str
    source_id: str
    domain: str                 # claim corroboration counts distinct domains
    stratum: str                # which stratum it constrains (the player)
    kind: str                   # "COUNT" | "SUM"
    value: float
