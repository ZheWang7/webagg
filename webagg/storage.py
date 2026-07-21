from sqlalchemy import (Column, String, Float, DateTime, Integer, JSON,
                        Boolean, ForeignKey, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class SourceRow(Base):
    __tablename__ = "sources"
    source_id = Column(String, primary_key=True)
    url = Column(String, nullable=False)
    domain = Column(String, index=True)
    fetch_time = Column(DateTime, nullable=False)
    publish_time = Column(DateTime)
    title = Column(String)
    main_text = Column(String)
    formulation_id = Column(String, index=True)
    source_class = Column(String, index=True)       # fragmentation.classify(): reg/vendor/news/...
    authority_chain_id = Column(String, index=True) # groups a filing with its amendments
    doc_type = Column(String)                       # "Form D" | "Form D/A" | ...
    identity_anchored = Column(Boolean, default=False)  # gates the qbar cap (paper 4.4)


class MentionRow(Base):
    __tablename__ = "mentions"
    mention_id = Column(String, primary_key=True)
    source_id = Column(String, ForeignKey("sources.source_id"), index=True)
    entity_surface = Column(String, index=True)
    record_kind = Column(String, index=True)
    attribute = Column(String, index=True)
    value = Column(String)
    passage = Column(String)
    extracted_at = Column(DateTime)
    t_asof = Column(DateTime)                       # valid time: when the fact HOLDS
    value_num = Column(Float)                       # canonical numeric, base units (USD)
    currency = Column(String)                       # ISO code; required for money attrs
    date_role = Column(String)                      # "announced" | "closed" | "filed"
    extractor_id = Column(String, default="A")      # "A" | "B" dual-extraction pass
    self_conf = Column(Float, default=0.5)          # extractor self-reported confidence
    validator_flags = Column(JSON, default=list)    # typed-validator flags (list[str])
    accepted = Column(Boolean, default=False, index=True)
    # indexed: ER/corroboration select only accepted mentions; the phi-audit
    # and conformal calibration read the rejected ones.


class ClaimRow(Base):
    """Mirrors type_defs.Claim: a statement about the WHOLE stratum
    (SUM / COUNT). Raw material for the checksum certificate
    (design paper §5.2, Def. 12 / Thm. 4)."""
    __tablename__ = "claims"
    claim_id = Column(String, primary_key=True)
    source_id = Column(String, ForeignKey("sources.source_id"), index=True)
    stratum_surface = Column(String, index=True)
    functional = Column(String)                     # "SUM" | "COUNT"
    attribute = Column(String, index=True)
    value_num = Column(Float)
    currency = Column(String)
    t_asof = Column(DateTime)                       # claims supersede along chains too
    scope = Column(String, default="")              # verbatim scope words, kept raw
    tolerance = Column(Float, default=0.0)          # implied precision: "$63M" -> 0.5e6
    passage = Column(String, default="")


class RejectedSourceRow(Base):
    """Raw material for the phi-audit (paper §7.1, layered reading defense).

    Sources the relevance filter rejected. We KEEP main_text because the
    audit re-reads a random sample of rejections to upper-bound the
    false-negative rate (Clopper-Pearson, audit.py). Deleting rejections
    would leave the completeness certificate with an unmeasurable term.
    """
    __tablename__ = "rejected_sources"
    source_id = Column(String, primary_key=True)
    url = Column(String)
    rejection_score = Column(Float)                 # filter score at rejection time
    main_text = Column(String)                      # keep it: the audit re-reads it
    audited = Column(Boolean, default=False)        # sampled by the audit yet?
    audit_verdict = Column(Boolean)                 # True = actually relevant (a FN)


class SupersessionRow(Base):
    """Structural amendment edges (design paper §4.3).

    One row per 'new document supersedes old document' event within an
    authority chain (e.g. Form D/A supersedes Form D). Corroboration walks
    these edges to adopt the NEWEST version and exclude stale echoes --
    they must never be counted as independent witnesses of the dead value.
    """
    __tablename__ = "supersessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chain_id = Column(String, index=True)           # = Source.authority_chain_id
    old_source_id = Column(String, index=True)
    new_source_id = Column(String, index=True)
    reason = Column(String)                         # form_amendment | self_correction | explicit


class MeasurementRow(Base):
    """One row per measurement event. The spine of evaluation.

    NEW (guide §4.2): the stratum column. The SIGMOD version certifies
    PER STRATUM (paper pitfall: never print a single global interval),
    so U_hat, psi, m0, kappa, f1/f2 ... are all logged with their stratum.
    Run-global metrics (e.g. token cost) leave it None.
    """
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)     # which experiment
    step = Column(Integer, index=True)      # agent step
    stratum = Column(String, index=True)    # NEW: most metrics are per-stratum
    metric = Column(String, index=True)     # 'U_hat','f1','f2','psi','m0','kappa',...
    value = Column(Float)
    extra = Column(JSON)                    # anything else worth keeping


class ResolvedRecordRow(Base):
    """Mirrors type_defs.ResolvedRecord: top of the provenance chain.
    attributes / contributing_mentions are stored as JSON blobs -- SQLite
    is per-run and this table is read back whole, never joined on."""
    __tablename__ = "resolved_records"
    record_id = Column(String, primary_key=True)
    entity_id = Column(String, index=True)          # THIS IS THE STRATUM key
    record_kind = Column(String, index=True)
    attributes = Column(JSON)                       # {attr: CorroboratedValue.model_dump()}
    contributing_mentions = Column(JSON)            # list[str] FK -> mentions


class FrontierFormulationRow(Base):
    __tablename__ = "formulations"
    formulation_id = Column(String, primary_key=True)
    run_id = Column(String, index=True)
    step = Column(Integer)                 # step at which it was added
    query = Column(String)
    expected_yield = Column(Float)
    issued = Column(Integer)               # 0 / 1
    realized_yield = Column(Integer)
    p_success = Column(Float)              # LLM estimate P(>=1 new record)
    yield_if_success = Column(Float)       # LLM estimate E[new | success]
    stratum = Column(String, index=True)   # None = generic formulation
    gap_directed = Column(Integer, default=0)   # 1 if from the claims engine

    #   TODO: To be added when their pydantic models are defined in later chapters:
    #   DerivationEdgeRow (corroboration chapter: copy/derivation edges)
    #   MatchDecisionRow  (ER chapter: pairwise match adjudications)

class StratumStateRow(Base):
    """One row per (run, stratum): the certificate's terms, recomputable from provenance
    but snapshotted here so a report never re-runs the loop."""
    __tablename__ = "stratum_states"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    stratum = Column(String, index=True)
    certified = Column(String)             # None | "checksum" | "registry"
    claimed_count = Column(Integer)        # cardinality brake (App. E), if any
    cert_kind = Column(String)             # "COUNT" | "SUM" for checksum certs
    cert_belief = Column(Float)            # b of the certifying claim; (1-b) -> interval
    cert_delta_plus = Column(Float)        # SUM value-gap bound Delta+_g (Thm 4a)
    n_records = Column(Integer)            # N_g at snapshot
    f1 = Column(Integer)                   # singletons
    f2 = Column(Integer)                   # doubletons
    u_hat = Column(Float)
    psi = Column(Float)
    step = Column(Integer)                 # loop step of the snapshot


def get_session(db_path: str):
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def load_sources(session) -> list:
    """Rehydrate every SourceRow back into a pydantic Source.

    Inverse of Source.to_row(). end_to_end (design doc Sec. 7.1) runs
    AFTER the frontier loop has persisted everything, so the DB is the
    single source of truth we reload from -- this is also what makes
    runs auditable/replayable (design doc Sec. 8, "data contracts").
    """
    from sqlalchemy import select
    from .type_defs import Source          # lazy: avoids circular import
    out = []
    for r in session.scalars(select(SourceRow)):
        out.append(Source(
            source_id=r.source_id,
            url=r.url,                     # str -> HttpUrl (pydantic validates)
            domain=r.domain,
            fetch_time=r.fetch_time,
            publish_time=r.publish_time,   # may be None; corroboration falls
            title=r.title,                 #   back to fetch_time (Def. 8 edges)
            main_text=r.main_text or "",
            formulation_id=r.formulation_id,
            # NEW: rehydrate the SIGMOD fields, else supersession chains and
            # the qbar-cap exemption silently vanish on reload.
            source_class=r.source_class,
            authority_chain_id=r.authority_chain_id,
            doc_type=r.doc_type,
            identity_anchored=bool(r.identity_anchored),
        ))                                 # raw_html intentionally dropped
    return out                             #   (impl guide pitfall 3)


def load_mentions(session, accepted_only: bool = False) -> list:
    """Rehydrate MentionRows back into pydantic Mentions.

    Mentions are the atoms of provenance (Def. 3 / type_defs docstring);
    ER clusters them (design doc Sec. 5) and corroboration groups them
    per value (Sec. 3), so both stages need the pydantic form.

    accepted_only=True returns only mentions that passed the conformal
    gate -- what ER and corroboration should consume once §6 is built.
    """
    from sqlalchemy import select
    from .type_defs import Mention         # lazy: avoids circular import
    q = select(MentionRow)
    if accepted_only:
        q = q.where(MentionRow.accepted.is_(True))
    return [Mention(
        mention_id=r.mention_id,
        source_id=r.source_id,             # FK back to its Source -- the
        entity_surface=r.entity_surface,   #   provenance handle we never lose
        record_kind=r.record_kind,
        attribute=r.attribute,
        value=r.value,
        passage=r.passage,
        extracted_at=r.extracted_at,
        # NEW: bi-temporal / typed / gate state survives the round trip.
        t_asof=r.t_asof,
        value_num=r.value_num,
        currency=r.currency,
        date_role=r.date_role,
        extractor_id=r.extractor_id or "A",
        self_conf=r.self_conf if r.self_conf is not None else 0.5,
        validator_flags=r.validator_flags or [],
        accepted=bool(r.accepted),
    ) for r in session.scalars(q)]


def load_claims(session) -> list:
    """Rehydrate ClaimRows into pydantic Claims (for the ClaimsEngine, §11).
    Same pattern as load_sources/load_mentions."""
    from sqlalchemy import select
    from .type_defs import Claim           # lazy: avoids circular import
    return [Claim(
        claim_id=r.claim_id,
        source_id=r.source_id,
        stratum_surface=r.stratum_surface,
        functional=r.functional,
        attribute=r.attribute,
        value_num=r.value_num,
        currency=r.currency,
        t_asof=r.t_asof,
        scope=r.scope or "",
        tolerance=r.tolerance or 0.0,
        passage=r.passage or "",
    ) for r in session.scalars(select(ClaimRow))]
