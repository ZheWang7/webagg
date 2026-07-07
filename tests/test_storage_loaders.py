from datetime import datetime

from webagg.storage import get_session, load_sources, load_mentions
from webagg.type_defs import Source, Mention


def test_source_and_mention_round_trip(tmp_path):
    # write via to_row(), read back via the loaders -- fields must survive
    session = get_session(str(tmp_path / "roundtrip.sqlite"))
    t = datetime(2026, 1, 8, 12, 0, 0)

    src = Source(source_id="abc123", url="https://www.sec.gov/filing.htm",
                 domain="www.sec.gov", fetch_time=t, publish_time=None,
                 title="8-K", main_text="Acme raised $40M Series B",
                 formulation_id="f1")
    m = Mention(mention_id="abc123:amount:deadbeef", source_id="abc123",
                entity_surface="Acme, Inc.", record_kind="funding_round",
                attribute="amount", value="40000000",
                passage="Acme raised $40M", extracted_at=t)
    session.add(src.to_row()); session.add(m.to_row()); session.commit()

    (s2,), (m2,) = load_sources(session), load_mentions(session)
    assert s2.source_id == src.source_id and s2.domain == "www.sec.gov"
    assert s2.publish_time is None and s2.main_text == src.main_text
    assert m2 == m                      # Mention round-trips exactly
