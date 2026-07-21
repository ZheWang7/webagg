"""Extraction: the four-stage gate (impl guide §6.1, paper §7.1).

The reader is the dominant error in deployment, so extraction is a gate,
not a single call:

    1. relevance phi      is_relevant() -> (bool, conf); False is LOGGED
                          (RejectedSourceRow), never silently dropped
    2. dual extraction    two readers with independent failure modes
                          (prompts A and B); agreement passes,
                          disagreement abstains
    3. typed validators   validators.validate_mention() -- deterministic,
                          free, run before any probability is spent
    4. conformal gate     calibration.ConformalGate.accept() -- accepted
                          mentions carry miscoverage <= delta_E (Prop. 2)

extract_certified() orchestrates all four and returns only mentions with
accepted=True, plus the Claims both readers emitted. Abstentions and
rejections are returned as counts (an abstention is a COST, not an error)
so the pipeline can log them to measurements.
"""
from .llm import call_llm
from .type_defs import Mention, Source, Claim
from .validators import validate_mention, ExtractionContext, Reject
from .calibration import ConformalGate
from .canonicalize import canonicalize_value
from . import config
from datetime import datetime
import dateparser

RELEVANCE_SYS = open("prompts/relevance.txt").read()
EXTRACT_SYS_A = open("prompts/extract.txt").read()
EXTRACT_SYS_B = open("prompts/extract_b.txt").read()


def is_relevant(source: Source, query: str) -> tuple[bool, float]:
    """Stage 1. NOW RETURNS (relevant, confidence) -- guide §6.1.

    The confidence is stored as rejection_score on RejectedSourceRow so
    the phi-audit can stratify its sample (near-threshold rejections are
    where false negatives live)."""
    user = f"QUERY:\n{query}\n\nDOCUMENT:\n{source.main_text[:8000]}"
    # cheap model: relevance is a high-volume yes/no (guide ch. 5)
    out = call_llm(system=RELEVANCE_SYS, user=user,
                   purpose="relevance")["payload"]
    return bool(out.get("relevant")), float(out.get("confidence", 0.5))


def _parse_t_asof(raw) -> datetime | None:
    """LLM-reported as-of date string -> UTC-naive datetime (or None)."""
    if not raw:
        return None
    dt = dateparser.parse(str(raw))
    return dt.replace(tzinfo=None) if dt else None


def extract_mentions(source: Source, query: str,
                     extractor_id: str = "A") -> tuple[list[Mention], list[Claim]]:
    """One reader pass (stage 2 primitive). NOW RETURNS (mentions, claims):
    the SIGMOD prompts emit whole-stratum Claims ("three rounds totaling
    $63M" -> COUNT + SUM) alongside per-record mentions."""
    system = EXTRACT_SYS_A if extractor_id == "A" else EXTRACT_SYS_B
    user = f"QUERY:\n{query}\n\nDOCUMENT (id={source.source_id}):\n{source.main_text[:12000]}"
    # strong model: structured extraction is where quality pays (guide ch. 5)
    payload = call_llm(system=system, user=user, max_tokens=4096,
                       model=config.MODEL_STRONG,
                       purpose=f"extraction_{extractor_id}")["payload"]
    mentions = []
    for m in payload.get("mentions", []):
        # BOUNDARY COERCION: the prompt says "canonicalize numbers to base
        # units", so the model often emits value as a JSON NUMBER
        # (500000000), while Mention.value is strictly typed str. Coerce
        # ONCE here -- the one place untrusted LLM JSON enters -- rather
        # than loosening the model's type. Coercing before make_id keeps
        # the identifier convention (guide §4.3) byte-identical for both
        # extractors regardless of which JSON type the model chose.
        val = str(m["value"])
        # ID convention (guide §4.3): ALWAYS via Mention.make_id -- now
        # passage-aware, so two same-valued rounds on one list page stay
        # two mentions (see make_id's deviation note).
        mentions.append(Mention(
            mention_id=Mention.make_id(source.source_id, m["entity_surface"],
                                       m["record_kind"], m["attribute"],
                                       val, extractor_id,
                                       passage=m["passage"]),
            source_id=source.source_id,
            entity_surface=m["entity_surface"],
            record_kind=m["record_kind"],
            attribute=m["attribute"],
            value=val,
            passage=m["passage"],
            extracted_at=datetime.utcnow(),
            # typed, bi-temporal fields (guide §4.1) now come from the prompt:
            t_asof=_parse_t_asof(m.get("t_asof")) or source.publish_time,
            currency=m.get("currency"),
            date_role=m.get("date_role"),
            self_conf=float(m.get("self_conf", 0.5)),
            extractor_id=extractor_id,
        ))
    # a model sometimes repeats the SAME assertion verbatim (identical
    # entity/kind/attribute/value/passage). Those share an id BY DESIGN --
    # they are duplicates, not two records -- so collapse them here rather
    # than crash the primary key at persist time.
    mentions = list({m.mention_id: m for m in mentions}.values())
    claims = []
    for c in payload.get("claims", []):
        claims.append(Claim(
            claim_id=Claim.make_id(source.source_id, c["functional"],
                                   c["stratum_surface"]),
            source_id=source.source_id,
            stratum_surface=c["stratum_surface"],
            functional=c["functional"],
            attribute=c["attribute"],
            value_num=float(c["value_num"]),
            currency=c.get("currency"),
            t_asof=_parse_t_asof(c.get("t_asof")) or source.publish_time,
            scope=c.get("scope", ""),
            tolerance=float(c.get("tolerance", 0.0)),
            passage=c.get("passage", ""),
        ))
    return mentions, claims


def _norm_surface(s: str) -> str:
    """Light normalization for A/B pairing ONLY: lowercase, strip punctuation
    and legal suffixes' commas ("Acme, Inc." == "Acme Inc"). Real aliasing
    ("Acme" vs "Acme Robotics") stays ER's job -- here both readers saw the
    SAME passage, so punctuation is the only legitimate variation."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _pair_key(m: Mention) -> tuple[str, str, str, str]:
    """A and B mentions AGREE when entity (normalized), record kind,
    attribute AND canonical value coincide. The value must be in the key:
    one (entity, kind, attribute) slot can legitimately hold several
    mentions per reader (e.g. a round amount AND a cumulative echo on the
    same page), so slot-only pairing mismatches them -- a bug caught by
    the ch. 3-6 integration test. Unpaired mentions simply abstain, which
    subsumes the old explicit disagreement check."""
    return (_norm_surface(m.entity_surface), m.record_kind, m.attribute,
            canonicalize_value(m.value))


def extract_certified(source: Source, query: str,
                      ctx: ExtractionContext | None = None,
                      gate: ConformalGate | None = None
                      ) -> tuple[list[Mention], list[Claim], dict]:
    """Stages 2-4 (guide §6.1): dual extraction -> validators -> gate.

    Returns (accepted_mentions, claims, info). info counts the casualties
    per stage -- the pipeline logs them as measurements. NOTE: the guide's
    stated contract is `-> (mentions, claims)`; the info dict is our
    addition so abstention (an observable cost, Prop. 2) actually gets
    observed. Disagreement handling: we ABSTAIN (the guide allows
    "adjudication or abstention"; an adjudicator is a later, optional
    upgrade -- abstention is the conservative choice and never injects a
    wrong value).
    """
    gate = gate or ConformalGate()          # unfitted -> bootstrap accept-all
    ment_a, claims_a = extract_mentions(source, query, extractor_id="A")
    ment_b, claims_b = extract_mentions(source, query, extractor_id="B")
    b_by_key: dict[tuple, list[Mention]] = {}
    for m in ment_b:                        # lists: duplicates are possible
        b_by_key.setdefault(_pair_key(m), []).append(m)

    accepted: list[Mention] = []
    info = {"n_a": len(ment_a), "n_b": len(ment_b), "agreed": 0,
            "disagreed": 0, "b_only": 0, "validator_rejects": 0,
            "gate_abstains": 0}

    for ma in ment_a:
        # -- stage 2: agreement = the OTHER reader independently produced
        # the same (entity, kind, attribute, canonical value).
        matches = b_by_key.get(_pair_key(ma))
        if not matches:
            info["disagreed"] += 1          # A-side abstention: no partner
            continue                        # (A/B failure modes differ, so
                                            # agreement is the evidence)
        mb = matches.pop()
        info["agreed"] += 1
        # the agreed mention proceeds as A's copy, carrying the MIN of the
        # two self-confidences (the cautious reader sets the score)
        ma.self_conf = min(ma.self_conf, mb.self_conf)

        # -- stage 3: typed validators (deterministic, free).
        try:
            ma = validate_mention(ma, ctx)
        except Reject as r:
            info["validator_rejects"] += 1
            ma.validator_flags = list(ma.validator_flags) + [f"reject:{r.reason}"]
            continue                        # never reaches corroboration; the
                                            # flag records WHY (audit material)
        # -- stage 4: conformal gate.
        if gate.accept(ma):
            ma.accepted = True
            accepted.append(ma)
        else:
            info["gate_abstains"] += 1

    info["b_only"] = sum(len(v) for v in b_by_key.values())  # B-side leftovers

    # Claims: dual-extracted like mentions -- keep those where A and B agree
    # on (stratum, functional) within tolerance; claim corroboration proper
    # is the claims-engine chapter.
    claims: list[Claim] = []
    b_claims = {(canonicalize_value(c.stratum_surface), c.functional): c
                for c in claims_b}
    for ca in claims_a:
        cb = b_claims.get((canonicalize_value(ca.stratum_surface), ca.functional))
        if cb and abs(ca.value_num - cb.value_num) <= max(ca.tolerance,
                                                          cb.tolerance,
                                                          0.005 * abs(cb.value_num)):
            claims.append(ca)
    return accepted, claims, info
