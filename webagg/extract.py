from .llm import call_llm
from .type_defs import Mention, Source
from datetime import datetime
import hashlib

RELEVANCE_SYS = open("prompts/relevance.txt").read()
EXTRACT_SYS = open("prompts/extract.txt").read()


def is_relevant(source: Source, query: str) -> bool:
    user = f"QUERY:\n{query}\n\nDOCUMENT:\n{source.main_text[:8000]}"
    return bool(call_llm(system=RELEVANCE_SYS, user=user)["payload"].get("relevant"))


def extract_mentions(source: Source, query: str) -> list[Mention]:
    user = f"QUERY:\n{query}\n\nDOCUMENT (id={source.source_id}):\n{source.main_text[:12000]}"
    payload = call_llm(system=EXTRACT_SYS, user=user, max_tokens=4096)["payload"]
    out = []
    for m in payload.get("mentions", []):
        value_hash = hashlib.sha256(m["value"].encode()).hexdigest()[:8]
        out.append(Mention(
            mention_id=f"{source.source_id}:{m['attribute']}:{value_hash}",
            source_id=source.source_id,
            entity_surface=m["entity_surface"],
            record_kind=m["record_kind"],
            attribute=m["attribute"],
            value=m["value"],
            passage=m["passage"],
            extracted_at=datetime.utcnow(),
        ))
    return out
