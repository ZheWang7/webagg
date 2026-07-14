from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional, Literal
import hashlib
from .storage import SourceRow, MentionRow


class Source(BaseModel):
    """A fetched web resource. One row per URL we visited.

    NEW in the SIGMOD version: the last four fields. They let later layers
    (fragmentation routing, supersession, the adversarial reliability cap)
    read what they need straight off the Source instead of re-deriving it.
    """
    source_id: str                      # hash of url + fetch_time
    url: HttpUrl
    domain: str                         # e.g. "sec.gov"
    fetch_time: datetime
    publish_time: Optional[datetime]    # parsed from HTML meta, may be None
    title: Optional[str]
    main_text: str                      # post-trafilatura extraction
    raw_html: Optional[str] = None      # keep for one run; drop for storage
    formulation_id: str                 # which search formulation produced this
    source_class: Optional[str] = None  # fragmentation.classify() label: "registry" / "vendor" / "news" / ...
                                        # Drives scan-vs-join routing (design paper App. D).
    authority_chain_id: Optional[str] = None    # Registry chain key, e.g. "edgar:0001234567:D" groups a Form D with its D/A amendments;
    doc_type: Optional[str] = None      # "Form D" | "Form D/A" | "annual_report"
    identity_anchored: bool = False     # True for registries, known publishers, the entity's own domain.


    @classmethod
    def make_id(cls, url: str, fetch_time: datetime) -> str:
        # Identifier convention (guide §4.3): first 16 hex of SHA-256
        # over "url|fetch_time".
        h = hashlib.sha256(f"{url}|{fetch_time.isoformat()}".encode()).hexdigest()
        return h[:16]

    def to_row(self):
        return SourceRow(
            source_id=self.source_id,
            url=str(self.url),          # HttpUrl -> str for the DB column
            domain=self.domain,
            fetch_time=self.fetch_time,
            publish_time=self.publish_time,
            title=self.title,
            main_text=self.main_text,
            formulation_id=self.formulation_id,
            # NEW columns (added to SourceRow in storage.py)
            source_class=self.source_class,
            authority_chain_id=self.authority_chain_id,
            doc_type=self.doc_type,
            identity_anchored=self.identity_anchored,
        )


class Mention(BaseModel):
    """One extracted attribute assertion from one source.
    The atom of provenance: every value we ever consider is a Mention.

    NEW in the SIGMOD version: the Mention is now BI-TEMPORAL and TYPED.
    Bi-temporal = it carries both when we read it (extracted_at) and when
    the fact is claimed to hold (t_asof). Supersession (design paper §4.3)
    compares t_asof along an authority chain, never extracted_at -- a fresh
    scrape of a stale page is still a stale fact.
    Typed = numeric values get a canonical numeric form + currency so that
    "$40M" and "40 million dollars" corroborate instead of competing.
    """
    mention_id: str
    source_id: str                      # FK -> Source (provenance link, never optional)
    entity_surface: str                 # e.g. "Acme, Inc."
    record_kind: str                    # e.g. "funding_round"
    attribute: str                      # e.g. "amount"
    value: str                          # raw extracted string, kept verbatim
    passage: str                        # the verbatim text supporting the claim
    extracted_at: datetime              # transaction time: when WE extracted it
    t_asof: Optional[datetime] = None   # Valid time: when the fact holds ("as of Dec 2024")
    value_num: Optional[float] = None
    currency: Optional[str] = None
    date_role: Optional[str] = None     # "announced" | "closed" | "filed"
    extractor_id: str = "A"             # "A" or "B": which of the two dual-extraction passes produced this.
    self_conf: float = 0.5         # extractor's self-reported confidence
    validator_flags: list[str] = Field(default_factory=list)    # Names of typed validators this mention failed/tripped (validators.py).
    accepted: bool = False              # Set by the split-conformal gate (guide §6).

    @classmethod
    def make_id(cls, source_id: str, attribute: str,
                value: str, extractor_id: str) -> str:
        # Identifier convention (guide §4.3):
        #   source_id:attribute:value_hash[:8]:extractor_id
        # The extractor suffix is what prevents the two dual-extraction
        # mentions of the same value from colliding (this fixes the
        # mention_id collision bug from the earlier implementation).
        vh = hashlib.sha256(value.encode()).hexdigest()[:8]
        return f"{source_id}:{attribute}:{vh}:{extractor_id}"

    def to_row(self):
        return MentionRow(
            mention_id=self.mention_id,
            source_id=self.source_id,
            entity_surface=self.entity_surface,
            record_kind=self.record_kind,
            attribute=self.attribute,
            value=self.value,
            passage=self.passage,
            extracted_at=self.extracted_at,
            # NEW columns (added to MentionRow in storage.py)
            t_asof=self.t_asof,
            value_num=self.value_num,
            currency=self.currency,
            date_role=self.date_role,
            extractor_id=self.extractor_id,
            self_conf=self.self_conf,
            validator_flags=self.validator_flags,
            accepted=self.accepted,
        )


class Claim(BaseModel):
    """NEW MODEL (SIGMOD §4.1). A statement about the WHOLE stratum, not
    about one record: "three rounds totaling $63M" yields TWO Claims
    (one COUNT, one SUM). Claims are the raw material of the checksum
    certificate (design paper §5.2, Definition 12 / Theorem 4): a stated
    aggregate lets us bound how much of the stratum we have actually found.
    """
    claim_id: str
    source_id: str                      # FK -> Source (provenance link)
    stratum_surface: str                # entity surface the claim scopes over
    functional: Literal["SUM", "COUNT"]
    attribute: str                      # e.g. "amount"
    value_num: float               # the claimed total/count
    currency: Optional[str] = None
    t_asof: Optional[datetime] = None   # "as of FY2025": claims supersede along the report chain
    scope: str = ""                     # Verbatim scope words ("equity only", "to date")
    tolerance: float = 0.0         # Implied by stated precision: "$63M" means +/- 0.5e6, not exact.
    passage: str = ""                   # verbatim supporting text

    def to_row(self):
        # Same persistence pattern as Source/Mention (ClaimRow in storage.py).
        from .storage import ClaimRow      # lazy: keeps import graph acyclic
        return ClaimRow(
            claim_id=self.claim_id,
            source_id=self.source_id,
            stratum_surface=self.stratum_surface,
            functional=self.functional,
            attribute=self.attribute,
            value_num=self.value_num,
            currency=self.currency,
            t_asof=self.t_asof,
            scope=self.scope,
            tolerance=self.tolerance,
            passage=self.passage,
        )


class ResolvedEntity(BaseModel):
    """An entity-resolution cluster: many surface forms -> one entity_id.
    (Kept from the original implementation; the entity_id is the STRATUM
    key used everywhere downstream.)"""
    entity_id: str                      # stable cluster id ("ent_seq", guide §4.3)
    canonical_name: str                 # representative surface form
    member_surfaces: list[str]
    confidence: float                   # cluster cohesion score in [0,1]


class CorroboratedValue(BaseModel):
    """A single adopted value with its belief and provenance.

    NEW in the SIGMOD version: supersession-awareness (version_id,
    n_dead_excluded) and the forgery margin kappa.
    """
    value: str
    belief: float                       # b(v*) from the design paper, in [0,1]
    nu: int                             # effective independence count
    component_sizes: list[int]          # |comp| for each component of G_v
    competing: dict[str, float] = Field(default_factory=dict)   # value -> belief

    # --- NEW (SIGMOD §4.1) ---
    value_num: Optional[float] = None   # canonical numeric form of the adopted value

    version_id: int = 0
    # Which supersession version was adopted (0 = no supersession seen).
    # Design paper §4.3: when a value is corrected, we adopt the newest
    # version along the authority chain, not the most-echoed one.

    n_dead_excluded: int = 0
    # How many mentions were excluded as stale echoes of superseded
    # versions. Kept as a count so the exclusion is auditable.

    kappa: Optional[int] = None
    # Forged-origin margin (design paper §4.5, Proposition 1): how many
    # independent witnesses an adversary would need to forge to flip the
    # adopted value. A DIAGNOSTIC only -- never enters interval arithmetic
    # (explicit pitfall in the paper).

    # --- Provenance discipline (guide §4.1 box) ---
    supporting_mention_ids: list[str] = Field(default_factory=list)
    # FK list -> Mention: exactly the mentions that asserted the ADOPTED
    # value. NOTE: this field is our addition, not in the guide's verbatim
    # model -- but the guide's discipline ("every object carries a foreign
    # key back to a Mention or Source") requires it, otherwise
    # CorroboratedValue is the one object that cannot be walked back.


class ResolvedRecord(BaseModel):
    """A logical record (e.g. one funding round) after ER + corroboration.
    Top of the provenance chain: ResolvedRecord -> Mentions -> Sources ->
    raw URL must be walkable in under thirty seconds (guide §4.1)."""
    record_id: str
    entity_id: str                      # FK -> ResolvedEntity; THIS IS THE STRATUM
    record_kind: str
    attributes: dict[str, CorroboratedValue]
    contributing_mentions: list[str]    # FK list -> Mention
