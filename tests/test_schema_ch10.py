"""SIGMOD guide Sec. 10 smoke tests -- one test per edit, fully OFFLINE.

No network, no API keys: EDGAR and ClinicalTrials are served by an
httpx.MockTransport, and the runner uses a fake in-memory driver.
Failures point at the responsible edit:

  test_edgar_stamps_registry_provenance -> EDGARDriver.fetch_for_key stamps
  test_stamps_exempt_qbar_cap           -> identity_anchored vs. QTable cap
  test_supersession_edge_for_free       -> chain + doc_type -> D/A supersedes D
  test_ctgov_refetch_supersedes         -> ClinicalTrialsDriver version chain
  test_blocking_predicate_logged        -> runner records K' predicate
  test_unblocked_sweep_not_flagged      -> full-K sweep stays blocked=False
  test_oracle_requires_deterministic_parser -> as_oracle refuses LLM default
  test_oracle_writes_truth_db           -> answer key lands in *_truth.sqlite

Run: pytest tests/test_schema_ch10.py -v
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webagg import config                                          # noqa: E402
from webagg.schema_addressable import (EDGARDriver,                # noqa: E402
                                       ClinicalTrialsDriver,
                                       run_schema_addressable)
from webagg.corroboration import (QTable, is_amendment,            # noqa: E402
                                  supersession_edges)
from webagg.type_defs import Source                                # noqa: E402
from webagg.storage import MeasurementRow                          # noqa: E402


# --------------------------------------------------------------------------- #
# Offline EDGAR: a MockTransport that impersonates sec.gov
# --------------------------------------------------------------------------- #
_CIK = "0000000123"

# One filer with a Form D and its later amendment (Form D/A). This is the
# minimal fixture that exercises BOTH halves of the Sec. 10 claim: the chain
# id groups them, and the amendment doc_type makes supersession fire.
_SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001-24-000001", "0001-24-000002"],
            "form": ["D", "D/A"],
            "filingDate": ["2024-01-10", "2024-03-05"],
            "primaryDocument": ["primary_doc.html", "primary_doc.html"],
        },
        "files": [],   # no shards for this tiny filer
    }
}

_FILING_HTML = ("<html><body>" + "Acme Corp raised money. " * 20
                + "</body></html>")


def _edgar_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("company_tickers.json"):
        return httpx.Response(200, json={
            "0": {"cik_str": 123, "ticker": "ACME", "title": "Acme Corp"}})
    if "data.sec.gov/submissions/" in url:
        return httpx.Response(200, json=_SUBMISSIONS)
    if "Archives/edgar/data" in url:
        return httpx.Response(200, text=_FILING_HTML)
    return httpx.Response(404)


def _edgar() -> EDGARDriver:
    client = httpx.Client(transport=httpx.MockTransport(_edgar_handler))
    return EDGARDriver(client=client, min_interval_s=0.0)


# --------------------------------------------------------------------------- #
# 1. The stamps exist and are correct
# --------------------------------------------------------------------------- #
def test_edgar_stamps_registry_provenance():
    srcs = _edgar().fetch_for_key(_CIK)
    assert len(srcs) == 2
    d, da = srcs                     # in filing order: Form D then Form D/A

    # doc_type carries the form verbatim ("/A" is what marks the amendment)
    assert d.doc_type == "Form D"
    assert da.doc_type == "Form D/A"
    # base filing and amendment share ONE authority chain (family "D")
    assert d.authority_chain_id == da.authority_chain_id == f"edgar:{_CIK}:D"
    # registry origin: anchored + classed, so priors/cap behave (next test)
    for s in srcs:
        assert s.identity_anchored is True
        assert s.source_class == "regulatory"
        assert s.domain == "sec.gov"


# --------------------------------------------------------------------------- #
# 2. identity_anchored lifts the adversarial q-bar cap
# --------------------------------------------------------------------------- #
def test_stamps_exempt_qbar_cap():
    src = _edgar().fetch_for_key(_CIK)[0]
    q = QTable()                     # fixed priors, qbar = 0.30
    # anchored registry source reads the full 0.95 regulatory prior...
    assert q.q(src) == pytest.approx(0.95)
    # ...whereas the SAME class WITHOUT the anchor is capped at q-bar: a
    # forged page merely CLAIMING to be regulatory buys at most 0.30.
    unanchored = Source(
        source_id="deadbeef00000000", url="https://sec-gov.example/fake",
        domain="sec-gov.example", fetch_time=datetime(2024, 1, 1),
        publish_time=None, title="fake", main_text="x" * 250,
        formulation_id="f0", source_class="regulatory",
        identity_anchored=False)
    assert q.q(unanchored) == pytest.approx(0.30)


# --------------------------------------------------------------------------- #
# 3. Supersession edges arrive "for free" from the stamps
# --------------------------------------------------------------------------- #
def test_supersession_edge_for_free():
    d, da = _edgar().fetch_for_key(_CIK)
    assert is_amendment(da.doc_type) and not is_amendment(d.doc_type)
    edges = supersession_edges([d, da])
    # exactly one structural edge: the D/A supersedes its Form D
    assert (d.source_id, da.source_id, "form_amendment") in edges
    assert all(e[2] != "form_amendment" or e[:2] == (d.source_id, da.source_id)
               for e in edges)


# --------------------------------------------------------------------------- #
# 4. ClinicalTrials: a later fetch of the same NCT supersedes the earlier one
# --------------------------------------------------------------------------- #
def _study_json(nct: str, last_update: str) -> dict:
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": "Trial " + nct},
        "statusModule": {
            "overallStatus": "RECRUITING",
            "studyFirstPostDateStruct": {"date": "2023-01-01"},
            "lastUpdatePostDateStruct": {"date": last_update},
        },
        "designModule": {"phases": ["PHASE3"],
                         "enrollmentInfo": {"count": 100}},
        "conditionsModule": {"conditions": ["X"]},
        "armsInterventionsModule": {"interventions": []},
        "descriptionModule": {"briefSummary": "s"},
    }}


def test_ctgov_refetch_supersedes():
    # stateful handler: the record mutates between the two fetches
    updates = iter(["2024-01-01", "2024-06-01"])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_study_json("NCT01", next(updates)))

    ct = ClinicalTrialsDriver(
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    (v1,) = ct.fetch_for_key("NCT01")
    (v2,) = ct.fetch_for_key("NCT01")

    # one chain per trial; every version is an "update" -> amendment-style
    assert v1.authority_chain_id == v2.authority_chain_id == "ctgov:NCT01"
    assert is_amendment(v1.doc_type) and is_amendment(v2.doc_type)
    # publish_time now dates the VERSION (last update), so v2 orders after v1
    assert v1.publish_time < v2.publish_time
    assert (v1.source_id, v2.source_id, "form_amendment") \
        in supersession_edges([v1, v2])


# --------------------------------------------------------------------------- #
# Runner tests: fake driver, no network, no LLM
# --------------------------------------------------------------------------- #
class _FakeDriver:
    name = "fake"

    def __init__(self, n: int = 3):
        self.n = n

    def enumerate_keys(self, query_filter):
        yield from (f"K{i}" for i in range(self.n))

    def fetch_for_key(self, key):
        now = datetime.utcnow()
        url = f"https://registry.example/{key}"
        return [Source(
            source_id=Source.make_id(url, now), url=url,
            domain="registry.example", fetch_time=now, publish_time=None,
            title=key, main_text="x" * 250, formulation_id=f"fake:{key}",
            source_class="regulatory", doc_type="registry record",
            authority_chain_id=f"fake:{key}", identity_anchored=True)]


def _fresh_run_id() -> str:
    return "test_ch10_" + uuid.uuid4().hex[:8]


def _cert_row(session):
    return (session.query(MeasurementRow)
            .filter(MeasurementRow.metric == "schema_complete").one())


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def test_blocking_predicate_logged():
    run_id = _fresh_run_id()
    out = run_schema_addressable(
        "q", _FakeDriver(), query_filter={"forms": ["D"], "since": "2024"},
        run_id=run_id, relevance_fn=lambda s, q: False,   # skip extraction
        extract_fn=lambda s, q: [], max_keys=2)
    try:
        assert out["blocked"] is True
        assert out["blocking_predicate"] == {
            "forms": ["D"], "since": "2024", "max_keys": 2}
        extra = _cert_row(out["session"]).extra
        # the certificate row itself records the predicate + delta_F = 0
        assert extra["blocked"] is True
        assert extra["blocking_predicate"]["forms"] == ["D"]
        assert extra["delta_F"] == 0.0 and extra["role"] == "agent"
        assert out["keys_swept"] == 2                     # max_keys honored
    finally:
        out["session"].close()
        _cleanup(out["db_path"])


def test_unblocked_sweep_not_flagged():
    run_id = _fresh_run_id()
    out = run_schema_addressable(
        "q", _FakeDriver(), query_filter={}, run_id=run_id,
        relevance_fn=lambda s, q: False, extract_fn=lambda s, q: [])
    try:
        assert out["blocked"] is False
        assert _cert_row(out["session"]).extra["blocked"] is False
        assert out["keys_swept"] == 3                     # all of K swept
    finally:
        out["session"].close()
        _cleanup(out["db_path"])


def test_oracle_requires_deterministic_parser():
    # the grader must never fall back to the student's LLM extractor
    with pytest.raises(ValueError):
        run_schema_addressable("q", _FakeDriver(), query_filter={},
                               run_id=_fresh_run_id(), as_oracle=True)


def test_oracle_writes_truth_db():
    run_id = _fresh_run_id()
    out = run_schema_addressable(
        "q", _FakeDriver(), query_filter={}, run_id=run_id,
        extract_fn=lambda s, q: [],       # stand-in deterministic parser
        as_oracle=True)
    try:
        # answer key in its own DB, tagged as the oracle's sweep
        assert out["db_path"].endswith(f"{run_id}_truth.sqlite")
        assert out["role"] == "oracle"
        assert _cert_row(out["session"]).extra["role"] == "oracle"
        # relevance defaulted to accept-all: every fetched doc was kept
        assert out["keys_swept"] == 3
    finally:
        out["session"].close()
        _cleanup(out["db_path"])
