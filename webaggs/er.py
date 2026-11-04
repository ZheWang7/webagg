"""Entity resolution for the citation domain (webaggr.pdf §6) — deterministic.

The stratum is the TARGET paper (one per run), and the "entity" a mention
asserts is which paper the page says is being cited. The guard: a mention
counts only when the page's cited-paper surface matches the target title —
exact normalized match, or one normalized title containing the other
(subtitle variants: 'Attention Is All You Need' vs the same plus venue
suffixes). Anything else is contamination (a citers-of-some-OTHER-paper
list) and is rejected, never merged. Citing-paper IDENTITY (the record
key) is handled by canon.cite_key_from_parts, not here.
"""
from __future__ import annotations

from .canon import normalize_title


def resolve_entity(entity_surface: str, target_paper: str) -> str | None:
    target = normalize_title(target_paper)
    surf = normalize_title(entity_surface)
    if not surf or not target:
        return None
    if surf == target:
        return target
    # subtitle/venue-suffix variants: containment with enough overlap
    shorter, longer = sorted((surf, target), key=len)
    if len(shorter.split()) >= 3 and shorter in longer:
        return target
    return None                                # different paper: contamination
