"""Certified reading: relevance gate + mention/claim extraction (webaggr.pdf §7.1).

Flow per source (impl guide §14.1 inner loop):
  is_relevant(src, player)  -> rejected sources stored for the phi-audit
  extract_certified(src)    -> LLM pulls (season, points) rows and aggregate
                               claims; typed validators canonicalize; the
                               conformal gate accepts/rejects each mention.
"""
from __future__ import annotations

from . import canon
from .llm import call_llm, load_prompt
from .types import Claim, Mention, sha16


def is_relevant(src, player_name: str) -> tuple[bool, float]:
    """Cheap relevance gate: the page must mention the player. (The LLM reader
    runs only on relevant pages; rejected pages feed the phi-audit, §7.1.)"""
    norm = canon.normalize_name(player_name)
    text_norm = canon.normalize_name(src.main_text[:40000])
    hit = norm in text_norm
    # last-name-only hit scores lower; used by the audit to prioritize re-reads
    last = norm.split()[-1] if norm else ""
    score = 1.0 if hit else (0.4 if last and last in text_norm else 0.0)
    return hit, score


def extract_certified(src, player_name: str, gate, *, ledger=None, run_id="",
                      llm=call_llm) -> tuple[list[Mention], list[Claim], int]:
    """Return (accepted mentions, claims, n_rejected_by_gate).

    `llm` is injectable for tests only — experiment paths always use the
    real call_llm (no fakes in any experiment path).
    """
    system = load_prompt("extract.txt")
    user = (f"Player: {player_name}\nPage text:\n{src.main_text[:12000]}")
    out = llm(system, user, ledger=ledger, run_id=run_id, purpose="extract")

    mentions: list[Mention] = []
    rejected = 0
    for row in out.get("season_rows", []):
        season = canon.canonical_season(str(row.get("season", "")))
        points = canon.canonical_points(row.get("points"))
        conf = float(row.get("confidence", 0.0))
        if season is None or points is None:
            rejected += 1                       # typed validator rejection
            continue
        if not gate.accepts(conf):
            rejected += 1                       # conformal rejection (delta_E)
            continue
        m = Mention(
            mention_id=f"{src.source_id}:points:{sha16(f'{season}|{points}')[:8]}",
            source_id=src.source_id, entity_surface=row.get("player", player_name),
            season=season, attribute="points", value=points, confidence=conf)
        mentions.append(m)

    claims: list[Claim] = []
    player_norm = canon.normalize_name(player_name)
    for c in out.get("claims", []):
        kind = str(c.get("kind", "")).upper()
        val = canon.canonical_points(c.get("value"))
        if kind not in ("COUNT", "SUM") or val is None:
            continue
        if kind == "COUNT" and not (1 <= val <= 30):
            continue                            # typed bound: NBA careers <= ~25 seasons
        claims.append(Claim(
            claim_id=f"{src.source_id}:CLAIM:{kind}:{sha16(str(val))[:8]}",
            source_id=src.source_id, domain=src.domain,
            stratum=player_norm, kind=kind, value=float(val)))
    return mentions, claims, rejected
