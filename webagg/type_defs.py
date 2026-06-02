from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional, Literal
import hashlib
from .storage import SourceRow, MentionRow


class Source(BaseModel):
    """A fetched web resource. One row per URL we visited."""
    source_id: str                      # hash of url + fetch_time
    url: HttpUrl
    domain: str                         # e.g. "sec.gov"
    fetch_time: datetime
    publish_time: Optional[datetime]    # parsed from HTML meta, may be None
    title: Optional[str]
    main_text: str                      # post-trafilatura extraction
    raw_html: Optional[str] = None      # keep for one run; drop for storage
    formulation_id: str                 # which search formulation produced this

    @classmethod
    def make_id(cls, url: str, fetch_time: datetime) -> str:
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
        )


class Mention(BaseModel):
    """One extracted attribute assertion from one source.
    The atom of provenance: every claim we ever consider is a Mention."""
    mention_id: str
    source_id: str                      # FK -> Source
    entity_surface: str                 # e.g. "Acme, Inc."
    record_kind: str                    # e.g. "funding_round"
    attribute: str                      # e.g. "amount"
    value: str                          # always store as string; cast on demand
    passage: str                        # the verbatim text supporting the claim
    extracted_at: datetime

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
        )


class ResolvedEntity(BaseModel):
    """An entity-resolution cluster: many surface forms -> one entity_id."""
    entity_id: str                      # stable cluster id
    canonical_name: str                 # representative surface form
    member_surfaces: list[str]
    confidence: float                   # cluster cohesion score in [0,1]


class ResolvedRecord(BaseModel):
    """A logical record (e.g. one funding round) after ER + corroboration."""
    record_id: str
    entity_id: str                      # FK -> ResolvedEntity
    record_kind: str
    attributes: dict[str, "CorroboratedValue"]
    contributing_mentions: list[str]    # FK list -> Mention


class CorroboratedValue(BaseModel):
    """A single adopted value with its belief and provenance."""
    value: str
    belief: float                       # b(v*) from the design doc, in [0,1]
    nu: int                             # effective independence count
    component_sizes: list[int]          # |comp| for each component of G_v
    competing: dict[str, float] = Field(default_factory=dict)   # value -> belief
