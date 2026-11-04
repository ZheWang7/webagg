"""Certified reading, citation domain (webaggr.pdf §7.1).

Per source: is_relevant gates on the page naming the TARGET paper; the LLM
reader pulls citing-paper rows (title, DOI when printed, confidence); typed
validators form the record key (DOI-first, canon.cite_key_from_parts); the
conformal gate prices extraction confidence. Aggregate claims are DISABLED
in this domain (config.CLAIMS_ENABLED, pre-registered): cited-by counts
differ across indexing universes, so a corroborated-but-wrong COUNT would
poison the cardinality brake; Exp 2-C tests the statistical path pure.
"""
from __future__ import annotations

from . import canon, config
from .llm import call_llm, load_prompt
from .types import Mention, sha16


def is_relevant(src, target_paper: str) -> tuple[bool, float]:
    """The page must name the target paper (first words of its normalized
    title). Rejected pages feed the phi-audit, as before."""
    target = canon.normalize_title(target_paper)
    probe = " ".join(target.split()[:6])          # leading words identify it
    text = canon.normalize_title(src.main_text[:40000])
    if probe and probe in text:
        return True, 1.0
    # weaker signal: more than half the title's words present
    words = [w for w in target.split() if len(w) > 3]
    hit = sum(1 for w in words if w in text)
    score = hit / max(len(words), 1)
    return score > 0.8, round(score, 2)


def extract_certified(src, target_paper: str, gate, *, ledger=None, run_id="",
                      llm=call_llm):
    """-> (accepted mentions, claims=[], n_rejected). Each accepted mention
    is one CITING paper: record key in the `season` slot (core modules and
    ledger schema unchanged — the slot is the record sub-key), value
    CITE_VALUE=1 so the unchanged SUM aggregate is the COUNT."""
    system = load_prompt("extract.txt")
    user = f"Target paper: {target_paper}\nPage text:\n{src.main_text[:12000]}"
    out = llm(system, user, ledger=ledger, run_id=run_id, purpose="extract")

    mentions, rejected = [], 0
    for row in out.get("citations", []):
        key = canon.cite_key_from_parts(row.get("doi"), row.get("title"))
        conf = float(row.get("confidence", 0.0))
        if key is None:
            rejected += 1                        # typed validator rejection
            continue
        if not gate.accepts(conf):
            rejected += 1                        # conformal rejection (delta_E)
            continue
        mentions.append(Mention(
            mention_id=f"{src.source_id}:cite:{sha16(key)[:8]}",
            source_id=src.source_id,
            entity_surface=row.get("cited_paper", target_paper),
            season=key, attribute="citation",
            value=config.CITE_VALUE, confidence=conf))
    return mentions, [], rejected                # claims OFF (pre-registered)
