"""Sec. 15 (withheld-registry oracle) -- the ground-truth builder, offline.

One test per duty (repo convention):
  test_parse_basic_fields                 -> XML tag lookups land in FormDFiling
  test_parse_indefinite_and_yet_to_occur  -> the two legal non-values handled
  test_parse_broken_xml_raises            -> a bad doc fails LOUDLY, never half-parses
  test_chain_collapse_amendment_supersedes-> D + D/A = ONE round, newest amount
  test_two_independent_rounds             -> unlinked Ds stay separate records
  test_partial_chain_flagged              -> D/A w/o its D still counts, flagged
  test_duplicate_truth_key_flagged        -> two rounds, one date -> both flagged
  test_truth_key_matches_default_truth_key-> oracle key == agent-side key, byte-equal
  test_truth_entity_sum_count             -> TruthEntity math over collapsed rounds
  test_json_roundtrip                     -> save -> load -> identical answer key
  test_split_deterministic_and_disjoint   -> cal/val halves reproducible, no leak
  test_oracle_sweep_end_to_end_offline    -> MockTransport EDGAR: xsl path
                                             stripped, RAW XML kept, truth values
                                             correct, oracle mentions stored
"""
from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from webagg import config
from webagg.formd import (FormDFiling, build_truth_entity, collapse_chains,
                          formd_mentions, load_truth_cohort,
                          load_truth_entity, parse_form_d, parse_source,
                          save_truth_entity)
from webagg.risk_control import default_truth_key, split_cohort
from webagg.schema_addressable import EDGARDriver, run_schema_addressable


# --------------------------------------------------------------------------- #
# XML fixtures: the OFFICIAL Form D primary_doc.xml shape. Keeping the tag
# paths honest here is the whole value of the suite -- the offline fake must
# pin the real contract (the gap_formulations() lesson).
# --------------------------------------------------------------------------- #
def _form_d_xml(*, name="Acme Robotics Inc.", cik="1234567",
                amendment=False, previous=None,
                first_sale="2024-01-10", yet_to_occur=False,
                sold="5000000", offered="8000000") -> str:
    amend_block = (
        f"<isAmendment>true</isAmendment>"
        f"<previousAccessionNumber>{previous}</previousAccessionNumber>"
        if amendment else "<isAmendment>false</isAmendment>")
    date_block = ("<yetToOccur>true</yetToOccur>" if yet_to_occur
                  else f"<value>{first_sale}</value>")
    return f"""<?xml version="1.0"?>
<edgarSubmission>
  <schemaVersion>X0708</schemaVersion>
  <submissionType>{'D/A' if amendment else 'D'}</submissionType>
  <primaryIssuer>
    <cik>{cik}</cik>
    <entityName>{name}</entityName>
  </primaryIssuer>
  <offeringData>
    <typeOfFiling>
      <newOrAmendment>{amend_block}</newOrAmendment>
      <dateOfFirstSale>{date_block}</dateOfFirstSale>
    </typeOfFiling>
    <offeringSalesAmounts>
      <totalOfferingAmount>{offered}</totalOfferingAmount>
      <totalAmountSold>{sold}</totalAmountSold>
      <totalRemaining>0</totalRemaining>
    </offeringSalesAmounts>
  </offeringData>
</edgarSubmission>"""


def _filing(acc, *, fdate, **kw) -> FormDFiling:
    return parse_form_d(_form_d_xml(**kw), accession=acc, filing_date=fdate)


# --------------------------------------------------------------------------- #
# 1-3. The parser
# --------------------------------------------------------------------------- #
def test_parse_basic_fields():
    f = _filing("0001234567-24-000001", fdate="2024-01-15")
    assert f.entity_name == "Acme Robotics Inc."
    assert f.cik == "0001234567"          # zero-padded to the EDGAR width
    assert f.is_amendment is False and f.previous_accession is None
    assert f.date_of_first_sale == "2024-01-10"
    assert f.amount_sold == 5_000_000.0
    assert f.offering_amount == 8_000_000.0
    assert f.filing_date == "2024-01-15"  # passed through from the index
    assert f.parse_flags == ()            # a clean doc raises no flags


def test_parse_indefinite_and_yet_to_occur():
    f = _filing("0001234567-24-000002", fdate="2024-02-01",
                yet_to_occur=True, sold="0", offered="Indefinite")
    assert f.date_of_first_sale is None   # no sale yet -> no date, no guess
    assert f.indefinite is True and f.offering_amount is None
    assert f.amount_sold == 0.0           # 0 sold is a NUMBER, not a failure
    assert "missing_date_of_first_sale" not in f.parse_flags  # yetToOccur set


def test_parse_broken_xml_raises():
    with pytest.raises(ET.ParseError):   # truncated doc -> loud, not partial
        parse_form_d("<edgarSubmission><primaryIssuer>",
                     accession="x", filing_date=None)


# --------------------------------------------------------------------------- #
# 4-7. Chain collapse
# --------------------------------------------------------------------------- #
def test_chain_collapse_amendment_supersedes():
    d = _filing("0001234567-24-000001", fdate="2024-01-15", sold="5000000")
    da = _filing("0001234567-24-000002", fdate="2024-03-01", amendment=True,
                 previous="0001234567-24-000001", sold="7500000")
    rounds = collapse_chains([d, da])
    assert len(rounds) == 1               # one chain = one round, COUNT=1
    r = rounds[0]
    assert r["amount"] == 7_500_000.0     # the D/A's number wins
    assert r["accession"] == "0001234567-24-000002"      # newest = authoritative
    assert r["root_accession"] == "0001234567-24-000001"
    assert r["n_filings"] == 2
    assert r["date"] == "2024-01-10"      # the round's date from the chain


def test_two_independent_rounds():
    a = _filing("0001234567-23-000001", fdate="2023-05-02", first_sale="2023-05-01",
                sold="2000000")
    b = _filing("0001234567-24-000001", fdate="2024-01-15", first_sale="2024-01-10",
                sold="5000000")
    rounds = collapse_chains([a, b])
    assert len(rounds) == 2               # no prev links -> two real rounds
    assert [r["amount"] for r in rounds] == [2_000_000.0, 5_000_000.0]


def test_partial_chain_flagged():
    # a D/A whose original D predates the sweep window: keep its (freshest)
    # numbers, but say so -- never silently drop registry truth
    da = _filing("0001234567-24-000009", fdate="2024-06-01", amendment=True,
                 previous="0001234567-19-000004", sold="9000000")
    rounds = collapse_chains([da])
    assert len(rounds) == 1
    assert rounds[0]["amount"] == 9_000_000.0
    assert "partial_chain" in rounds[0]["flags"]
    assert rounds[0]["root_accession"] == "0001234567-19-000004"  # stable root name


def test_duplicate_truth_key_flagged():
    # two DISTINCT rounds with the same first-sale date collide under the
    # kind|date key -- grading must know
    a = _filing("0001234567-24-000001", fdate="2024-01-15", first_sale="2024-01-10",
                sold="1000000")
    b = _filing("0001234567-24-000005", fdate="2024-02-20", first_sale="2024-01-10",
                sold="3000000")
    rounds = collapse_chains([a, b])
    assert len(rounds) == 2               # they stay separate records
    assert all("duplicate_truth_key" in r["flags"] for r in rounds)


# --------------------------------------------------------------------------- #
# 8. The key convention is LOCKED to the agent side
# --------------------------------------------------------------------------- #
def test_truth_key_matches_default_truth_key():
    d = _filing("0001234567-24-000001", fdate="2024-01-15", first_sale="2024-01-10")
    truth, _ = build_truth_entity("cik0001234567", [d])

    # a minimal stand-in for the agent's ResolvedRecord: record_kind +
    # a date attribute whose .value is the ISO date the pipeline stores
    agent_record = SimpleNamespace(
        record_kind="funding_round",
        attributes={"date": SimpleNamespace(value="2024-01-10")})
    assert default_truth_key(agent_record) == truth.records[0].key
    assert truth.records[0].key == "funding_round|2024-01-10"


# --------------------------------------------------------------------------- #
# 9-10. TruthEntity math + disk round-trip
# --------------------------------------------------------------------------- #
def test_truth_entity_sum_count():
    d1 = _filing("0001234567-23-000001", fdate="2023-05-02", first_sale="2023-05-01",
                 sold="2000000")
    d2 = _filing("0001234567-24-000001", fdate="2024-01-15", first_sale="2024-01-10",
                 sold="5000000")
    da2 = _filing("0001234567-24-000002", fdate="2024-03-01", amendment=True,
                  previous="0001234567-24-000001", sold="7500000")
    truth, meta = build_truth_entity("cik0001234567", [d1, d2, da2])
    assert truth.true_count == 2          # 3 filings, 2 rounds
    assert truth.true_sum == 9_500_000.0  # 2M + 7.5M (amended, not 5M)
    assert meta["n_filings"] == 3 and meta["n_rounds"] == 2


def test_json_roundtrip(tmp_path: Path):
    d = _filing("0001234567-24-000001", fdate="2024-01-15", sold="5000000")
    truth, meta = build_truth_entity("cik0001234567", [d])
    p = save_truth_entity(tmp_path, meta)
    truth2, meta2 = load_truth_entity(p)
    assert truth2 == truth                # frozen dataclasses: deep equality
    assert meta2 == meta
    cohort = load_truth_cohort(tmp_path)
    assert cohort == {"cik0001234567": truth}


# --------------------------------------------------------------------------- #
# 11. The cal/val split: reproducible, disjoint, leak-free
# --------------------------------------------------------------------------- #
def test_split_deterministic_and_disjoint():
    ids = [f"cik{i:010d}" for i in range(9)]
    cal1, val1 = split_cohort(ids, seed=7)
    cal2, val2 = split_cohort(ids, seed=7)
    assert (cal1, val1) == (cal2, val2)   # same seed -> same halves
    assert not (set(cal1) & set(val1))    # no entity on both sides
    assert sorted(cal1 + val1) == sorted(ids)   # nobody dropped


# --------------------------------------------------------------------------- #
# 12. The oracle sweep, end to end, against a MockTransport EDGAR
# --------------------------------------------------------------------------- #
_CIK = "0000000123"
_XML_D = _form_d_xml(name="Acme Robotics Inc.", cik="123",
                     first_sale="2024-01-10", sold="5000000")
_XML_DA = _form_d_xml(name="Acme Robotics Inc.", cik="123", amendment=True,
                      previous="0001234567-24-000001", first_sale="2024-01-10",
                      sold="7500000")
_SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002"],
            "form": ["D", "D/A"],
            "filingDate": ["2024-01-15", "2024-03-01"],
            # the REAL index lists Form D primaries behind the XSL rendering
            # path; the driver must strip it to reach the raw XML
            "primaryDocument": ["xslFormDX01/primary_doc.xml",
                                "xslFormDX01/primary_doc.xml"],
        },
        "files": [],
    }
}
_SERVED: list[str] = []                   # every archive URL the driver asked for


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "data.sec.gov/submissions/" in url:
        return httpx.Response(200, json=_SUBMISSIONS)
    if "Archives/edgar/data" in url:
        _SERVED.append(url)
        # serve the D or the D/A by accession segment in the path
        xml = _XML_D if "000123456724000001" in url else _XML_DA
        return httpx.Response(200, text=xml)
    return httpx.Response(404)


def test_oracle_sweep_end_to_end_offline():
    _SERVED.clear()
    driver = EDGARDriver(
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
        min_interval_s=0.0)
    run_id = f"t15_{uuid.uuid4().hex[:8]}"

    filings, mentions_seen = [], []

    def oracle_extract(src, query):       # same double-duty closure as the CLI
        filings.append(parse_source(src))
        ms = formd_mentions(src, query)
        mentions_seen.extend(ms)
        return ms

    out = run_schema_addressable(
        "form d funding rounds", driver,
        query_filter={"ciks": [_CIK], "forms": ["D", "D/A"]},
        run_id=run_id, extract_fn=oracle_extract, as_oracle=True)
    try:
        # the driver reached the RAW XML, not the HTML rendering
        assert _SERVED and all("xslFormD" not in u for u in _SERVED)

        # raw XML survived into Source.main_text (tag names intact)
        src_dump = filings[0]             # parse succeeded => tags were there
        assert src_dump.amount_sold == 5_000_000.0

        # the oracle's mentions are deterministic-grade provenance
        assert mentions_seen, "oracle sweep produced no mentions"
        assert all(m.extractor_id == "oracle" and m.accepted
                   and m.self_conf == 1.0 for m in mentions_seen)
        assert all("gate_uncalibrated" not in m.validator_flags
                   for m in mentions_seen)   # no gate ran, no bootstrap stamp

        # the answer key: chains collapsed, D/A's number wins
        truth, meta = build_truth_entity(f"cik{_CIK}", filings)
        assert truth.true_count == 1
        assert truth.true_sum == 7_500_000.0
        assert truth.records[0].key == "funding_round|2024-01-10"

        # and it went to the ORACLE's separate database
        assert out["db_path"].endswith(f"{run_id}_truth.sqlite")
        assert out["role"] == "oracle"
    finally:
        # Windows-safe teardown: closing the SESSION is not enough -- the
        # engine's connection pool still holds the SQLite file open, and
        # Windows (unlike POSIX) refuses to delete an open file. Grab the
        # engine, close, dispose (releases the handle), THEN unlink. The
        # unlink stays tolerant as a last resort: leftover temp DBs are
        # harmless residue, and cleanup must never be what fails a test.
        engine = out["session"].get_bind()
        out["session"].close()
        engine.dispose()
        try:
            Path(out["db_path"]).unlink(missing_ok=True)
        except OSError:
            pass
