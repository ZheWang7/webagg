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
