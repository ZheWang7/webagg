"""Smoke test for guide §4.1 (core models) and §4.2 (storage).

Run from the repo root:  pytest tests/test_smoke_4_1_4_2.py -v

Each test targets ONE edit from the update, so a failure points straight
at the file that still needs attention:
  - type_defs.py   (§4.1 models, make_id conventions)
  - storage.py     (§4.2 columns, new tables, rehydration helpers)
  - metrics.py     (stratum-aware log_measurement)
  - corroboration.py (provenance discipline: supporting_mention_ids)
"""
from datetime import datetime, timedelta

import pytest

from webagg.type_defs import Source, Mention, Claim, CorroboratedValue
from webagg.storage import (get_session, load_sources, load_mentions,
                            load_claims, RejectedSourceRow, SupersessionRow,
                            MeasurementRow)
from webagg.metrics import log_measurement

# PROJECT CONVENTION: all datetimes are UTC-NAIVE.
# SQLite's DateTime column drops tzinfo, so tz-aware datetimes come back
# naive after a round trip -- and Python raises TypeError when comparing
# aware vs naive, which would break derivation_edge() and supersession
# (both compare publish/fetch/t_asof times). Normalize at the boundary.
T0 = datetime(2026, 1, 15, 12, 0)                 # fixed: deterministic ids


@pytest.fixture()
def session(tmp_path):
    # Fresh per-test DB, like the per-run DBs in real use (guide §4.2).
    return get_session(str(tmp_path / "smoke.sqlite"))


def make_source(url="https://sec.gov/formd/acme", domain="sec.gov", **kw):
    """A registry source with all §4.1 fields set."""
    defaults = dict(
        source_id=Source.make_id(url, T0), url=url, domain=domain,
        fetch_time=T0, publish_time=T0 - timedelta(days=1),
        title="Form D", main_text="Acme raised money.", formulation_id="f1",
        source_class="registry", authority_chain_id="edgar:0001234567:D",
        doc_type="Form D", identity_anchored=True,
    )
    defaults.update(kw)
    return Source(**defaults)


def make_mention(src, value="$40M", extractor="A", **kw):
    """A typed, bi-temporal money mention (§4.1)."""
    defaults = dict(
        mention_id=Mention.make_id(src.source_id, "Acme, Inc.",
                                   "funding_round", "amount", value, extractor),
        source_id=src.source_id, entity_surface="Acme, Inc.",
        record_kind="funding_round", attribute="amount", value=value,
        passage=f"Acme raised {value} in its Series A.", extracted_at=T0,
        t_asof=T0 - timedelta(days=30), value_num=40e6, currency="USD",
        date_role="announced", extractor_id=extractor, self_conf=0.9,
        accepted=True,
    )
    defaults.update(kw)
    return Mention(**defaults)


# ---------------------------------------------------------------- §4.1 / §4.3
def test_id_conventions():
    # source_id: first 16 hex of SHA-256 over "url|fetch_time" -- deterministic.
    sid = Source.make_id("https://sec.gov/x", T0)
    assert sid == Source.make_id("https://sec.gov/x", T0) and len(sid) == 16

    # mention_id: extractor suffix keeps dual-extraction passes distinct
    # (this is the fix for the old mention_id collision bug).
    a = Mention.make_id(sid, "Acme, Inc.", "funding_round", "amount", "$40M", "A")
    b = Mention.make_id(sid, "Acme, Inc.", "funding_round", "amount", "$40M", "B")
    assert a != b and a.endswith(":A") and b.endswith(":B")
    assert a.startswith(f"{sid}:amount:")

    # entity-aware hash: two entities sharing attribute+value on ONE page
    # (list-page case) must not collide.
    acme = Mention.make_id(sid, "Acme, Inc.", "funding_round", "round_type", "Series A", "A")
    bolt = Mention.make_id(sid, "Bolt Ltd.", "funding_round", "round_type", "Series A", "A")
    assert acme != bolt

    # claim_id: source_id:CLAIM:functional:stratum_hash[:8] -- SUM and COUNT
    # claims from the same sentence get distinct ids.
    cs = Claim.make_id(sid, "SUM", "Acme, Inc.")
    cc = Claim.make_id(sid, "COUNT", "Acme, Inc.")
    assert cs != cc and cs.startswith(f"{sid}:CLAIM:SUM:") and len(cs.split(":")[-1]) == 8


def test_new_model_fields_exist():
    # §4.1 additions on CorroboratedValue, incl. our provenance field.
    cv = CorroboratedValue(value="$40M", belief=0.9, nu=2, component_sizes=[1, 1])
    assert cv.version_id == 0 and cv.n_dead_excluded == 0 and cv.kappa is None
    assert cv.supporting_mention_ids == []          # provenance discipline field
    # Claim is a whole-stratum statement (SUM/COUNT only).
    with pytest.raises(Exception):
        Claim(claim_id="c", source_id="s", stratum_surface="Acme",
              functional="AVG", attribute="amount", value_num=1.0)


# --------------------------------------------------------------------- §4.2
def test_source_roundtrip(session):
    src = make_source()
    session.add(src.to_row()); session.commit()
    back = load_sources(session)[0]
    # The four new columns must survive write AND rehydration.
    assert back.source_class == "registry"
    assert back.authority_chain_id == "edgar:0001234567:D"
    assert back.doc_type == "Form D"
    assert back.identity_anchored is True


def test_mention_roundtrip_and_accepted_filter(session):
    src = make_source()
    ok = make_mention(src, extractor="A", accepted=True,
                      validator_flags=["currency_ok"])
    rej = make_mention(src, extractor="B", accepted=False)
    session.add_all([src.to_row(), ok.to_row(), rej.to_row()]); session.commit()

    back = {m.mention_id: m for m in load_mentions(session)}
    m = back[ok.mention_id]
    # Bi-temporal + typed + gate state survives the round trip.
    assert m.t_asof == ok.t_asof and m.value_num == 40e6
    assert m.currency == "USD" and m.date_role == "announced"
    assert m.extractor_id == "A" and m.self_conf == 0.9
    assert m.validator_flags == ["currency_ok"] and m.accepted is True

    # accepted_only: what ER/corroboration will consume after the §6 gate.
    only = load_mentions(session, accepted_only=True)
    assert [x.mention_id for x in only] == [ok.mention_id]


def test_claim_roundtrip(session):
    src = make_source()
    c = Claim(claim_id=Claim.make_id(src.source_id, "SUM", "Acme, Inc."),
              source_id=src.source_id, stratum_surface="Acme, Inc.",
              functional="SUM", attribute="amount", value_num=63e6,
              currency="USD", scope="to date", tolerance=0.5e6,
              passage="three rounds totaling $63M to date")
    session.add_all([src.to_row(), c.to_row()]); session.commit()
    back = load_claims(session)[0]
    assert back.functional == "SUM" and back.value_num == 63e6
    assert back.scope == "to date" and back.tolerance == 0.5e6


def test_phi_audit_and_supersession_tables(session):
    # rejected_sources keeps main_text: the phi-audit re-reads it (§7.1).
    session.add(RejectedSourceRow(source_id="r1", url="https://blog.example/x",
                                  rejection_score=0.12,
                                  main_text="kept so the audit can re-read"))
    # supersessions: D/A amends D within one authority chain (paper §4.3).
    session.add(SupersessionRow(chain_id="edgar:0001234567:D",
                                old_source_id="old", new_source_id="new",
                                reason="form_amendment"))
    session.commit()
    r = session.query(RejectedSourceRow).one()
    assert r.audited is False and r.main_text     # text retained, not yet audited
    e = session.query(SupersessionRow).one()
    assert (e.chain_id, e.reason) == ("edgar:0001234567:D", "form_amendment")


def test_measurement_stratum(session):
    # Per-stratum metric (the SIGMOD way) and run-global metric (stratum=None)
    log_measurement(session, "run1", 3, "U_hat", 21.0, stratum="ent_0")
    log_measurement(session, "run1", 3, "token_cost", 1234)   # old call style still works
    session.commit()
    rows = {r.metric: r for r in session.query(MeasurementRow)}
    assert rows["U_hat"].stratum == "ent_0"
    assert rows["token_cost"].stratum is None


# ------------------------------------------------ provenance discipline (§4.1)
def test_provenance_walkback(session):
    """The 30-second walk: CorroboratedValue -> Mentions -> Sources -> URL."""
    from webagg.corroboration import corroborate

    # Two INDEPENDENT witnesses for $40M (different domains, different
    # passages, no derivation edge), one witness for the wrong $50M.
    s1 = make_source("https://sec.gov/formd/acme", "sec.gov")
    s2 = make_source("https://reuters.com/acme-round", "reuters.com",
                     source_class="news", authority_chain_id=None,
                     identity_anchored=False,
                     main_text="Financial newswire coverage of the round.")
    s3 = make_source("https://techcrunch.com/acme", "techcrunch.com",
                     source_class="news", authority_chain_id=None,
                     identity_anchored=False,
                     main_text="A startup blog post about Acme.")
    m1 = make_mention(s1, "$40M", "A",
                      passage="Form D discloses a $40M sale of securities.")
    m2 = make_mention(s2, "$40M", "A",
                      passage="Sources familiar with the deal said forty million.")
    m3 = make_mention(s3, "$50M", "A", value_num=50e6,
                      passage="Acme is rumored to have raised $50M.")

    sources = {s.source_id: s for s in (s1, s2, s3)}
    cv = corroborate({"$40M": [m1, m2], "$50M": [m3]}, sources)

    assert cv.value == "$40M" and cv.nu == 2        # two independent witnesses win
    # THE discipline: the adopted value knows exactly who asserted it...
    assert sorted(cv.supporting_mention_ids) == sorted([m1.mention_id, m2.mention_id])

    # ...and the chain walks back to raw URLs through the DB alone.
    session.add_all([s.to_row() for s in (s1, s2, s3)])
    session.add_all([m.to_row() for m in (m1, m2, m3)]); session.commit()
    mentions = {m.mention_id: m for m in load_mentions(session)}
    srcs = {s.source_id: s for s in load_sources(session)}
    urls = {str(srcs[mentions[mid].source_id].url)
            for mid in cv.supporting_mention_ids}
    assert urls == {"https://sec.gov/formd/acme", "https://reuters.com/acme-round"}
