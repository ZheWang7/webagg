from sqlalchemy import (Column, String, Float, DateTime, Integer, JSON,
                        ForeignKey, create_engine)
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


class MeasurementRow(Base):
    """One row per measurement event. The spine of evaluation."""
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)     # which experiment
    step = Column(Integer, index=True)      # agent step
    metric = Column(String, index=True)     # 'U_hat', 'r_t', 'N_t', ...
    value = Column(Float)
    extra = Column(JSON)                    # anything else worth keeping


# To be added in later sections, mirroring the matching pydantic models:
#   FrontierFormulationRow, ResolvedEntityRow, ResolvedRecordRow,
#   DerivationEdgeRow, MatchDecisionRow


class FrontierFormulationRow(Base):
    __tablename__ = "formulations"
    formulation_id = Column(String, primary_key=True)
    run_id = Column(String, index=True)
    step = Column(Integer)                 # step at which it was added
    query = Column(String)
    expected_yield = Column(Float)
    issued = Column(Integer)               # 0 / 1
    realized_yield = Column(Integer)


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
        ))                                 # raw_html intentionally dropped
    return out                             #   (impl guide pitfall 3)


def load_mentions(session) -> list:
    """Rehydrate every MentionRow back into a pydantic Mention.

    Mentions are the atoms of provenance (Def. 3 / type_defs docstring);
    ER clusters them (design doc Sec. 5) and corroboration groups them
    per value (Sec. 3), so both stages need the pydantic form.
    """
    from sqlalchemy import select
    from .type_defs import Mention         # lazy: avoids circular import
    return [Mention(
        mention_id=r.mention_id,
        source_id=r.source_id,             # FK back to its Source -- the
        entity_surface=r.entity_surface,   #   provenance handle we never lose
        record_kind=r.record_kind,
        attribute=r.attribute,
        value=r.value,
        passage=r.passage,
        extracted_at=r.extracted_at,
    ) for r in session.scalars(select(MentionRow))]
