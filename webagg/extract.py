from .llm import call_llm
from .type_defs import Mention, Source
from datetime import datetime
import hashlib

RELEVANCE_SYS = open("prompts/relevance.txt").read()
EXTRACT_SYS = open("prompts/extract.txt").read()


def is_relevant(source: Source, query: str) -> bool:
    user = f"QUERY:\n{query}\n\nDOCUMENT:\n{source.main_text[:8000]}"
    return bool(call_llm(system=RELEVANCE_SYS, user=user)["payload"].get("relevant"))


def extract_mentions(source: Source, query: str,
                     extractor_id: str = "A") -> list[Mention]:
    user = f"QUERY:\n{query}\n\nDOCUMENT (id={source.source_id}):\n{source.main_text[:12000]}"
    payload = call_llm(system=EXTRACT_SYS, user=user, max_tokens=4096)["payload"]
    out = []
    for m in payload.get("mentions", []):
        # ID convention (guide §4.3): ALWAYS via Mention.make_id -- never
        # hand-format. The entity-aware hash (distinct IDs for two entities
        # sharing attribute+value on one page) now lives inside make_id.
        out.append(Mention(
            mention_id=Mention.make_id(source.source_id, m["entity_surface"],
                                       m["record_kind"], m["attribute"],
                                       m["value"], extractor_id),
            source_id=source.source_id,
            entity_surface=m["entity_surface"],
            record_kind=m["record_kind"],
            attribute=m["attribute"],
            value=m["value"],
            passage=m["passage"],
            extracted_at=datetime.utcnow(),
            extractor_id=extractor_id,     # which dual-extraction pass (A/B)
        ))
    return out
