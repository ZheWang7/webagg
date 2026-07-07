"""Section 9 sanity tests for schema-addressable mode.

All offline: an httpx.MockTransport serves fixtures shaped like the real SEC and
ClinicalTrials.gov payloads, and the runner gets stubbed relevance/extract
functions, so there is NO network and NO LLM/API key needed. Run from repo root:
    pytest tests/test_schema_addressable.py
"""
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import httpx

from webagg.type_defs import Mention
from webagg.schema_addressable import (
    EDGARDriver, ClinicalTrialsDriver, run_schema_addressable,
)


# --------------------------------------------------------------------------- #
# Fixtures shaped like the real APIs
# --------------------------------------------------------------------------- #
TICKERS = {
    "0": {"cik_str": 1318605, "ticker": "TSLA", "title": "Tesla, Inc."},
    "1": {"cik_str": 320193,  "ticker": "AAPL", "title": "Apple Inc."},
}
# Tesla: 'recent' holds a 10-K + an 8-K; an OLDER 10-K is only in a shard file.
TSLA_SUBM = {
    "cik": "1318605", "name": "Tesla, Inc.",
    "filings": {
        "recent": {
            "accessionNumber": ["0001318605-23-000033", "0001318605-23-000050"],
            "form":            ["10-K",                 "8-K"],
            "filingDate":      ["2023-01-31",           "2023-04-19"],
            "primaryDocument": ["tsla-20221231.htm",    "ex991.htm"],
        },
        "files": [{"name": "CIK0001318605-submissions-001.json"}],
    },
}
TSLA_SHARD = {
    "accessionNumber": ["0001193125-11-000001"],
    "form":            ["10-K"],
    "filingDate":      ["2011-02-28"],
    "primaryDocument": ["d10k.htm"],
}
DOC_HTML = "<html><body><p>" + ("R&D expenses were $3,075 million. " * 30) + "</p></body></html>"


def _study(nct, enroll):
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": f"Study {nct}"},
        "statusModule": {"overallStatus": "COMPLETED",
                         "studyFirstPostDateStruct": {"date": "2018-05-01"}},
        "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": enroll}},
        "conditionsModule": {"conditions": ["Melanoma"]},
        "armsInterventionsModule": {"interventions": [
            {"type": "DRUG", "name": "Pembrolizumab"}]},
        "descriptionModule": {"briefSummary": "A phase 3 trial."},
    }}


CT_PAGE1 = {"studies": [_study("NCT00000001", 100), _study("NCT00000002", 250)],
            "nextPageToken": "PAGE2"}
CT_PAGE2 = {"studies": [_study("NCT00000003", 500)]}  # no token -> last page


def _handler(request: httpx.Request) -> httpx.Response:
    u = str(request.url)
    p = urlparse(u)
    if u == EDGARDriver.TICKERS_URL:
        return httpx.Response(200, json=TICKERS)
    if u == EDGARDriver.SUBMISSIONS_URL.format(cik="0001318605"):
        return httpx.Response(200, json=TSLA_SUBM)
    if u == EDGARDriver.SHARD_URL.format(name="CIK0001318605-submissions-001.json"):
        return httpx.Response(200, json=TSLA_SHARD)
    if "/Archives/edgar/data/" in u:
        return httpx.Response(200, text=DOC_HTML,
                              headers={"content-type": "text/html"})
    if p.path.endswith("/api/v2/studies"):
        token = parse_qs(p.query).get("pageToken", [None])[0]
        return httpx.Response(200, json=(CT_PAGE2 if token == "PAGE2" else CT_PAGE1))
    if "/api/v2/studies/" in u:
        nct = u.rstrip("/").split("/")[-1]
        return httpx.Response(200, json=_study(nct, 123))
    return httpx.Response(404, text=f"unmocked: {u}")


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_handler))


def _edgar() -> EDGARDriver:
    return EDGARDriver(client=_client(), min_interval_s=0.0)


def _ctgov() -> ClinicalTrialsDriver:
    return ClinicalTrialsDriver(client=_client(), min_interval_s=0.0)


# --------------------------------------------------------------------------- #
# EDGAR
# --------------------------------------------------------------------------- #
def test_edgar_enumerate_name_contains():
    assert list(_edgar().enumerate_keys({"name_contains": "tesla"})) == ["0001318605"]


def test_edgar_enumerate_ticker_exact():
    assert list(_edgar().enumerate_keys({"tickers": ["AAPL"]})) == ["0000320193"]


def test_edgar_explicit_ciks_skip_index():
    # explicit CIKs are zero-padded and used directly (no tickers fetch)
    assert list(_edgar().enumerate_keys({"ciks": ["320193"]})) == ["0000320193"]


def test_edgar_paginates_across_shards():
    """The OLDER 10-K lives only in a shard file, not in filings.recent.
    If shard pagination is broken we'd get 1 filing instead of 2."""
    d = _edgar()
    d.enumerate_keys({"name_contains": "tesla", "forms": ["10-K"], "since": "2010"})
    srcs = d.fetch_for_key("1318605")
    assert len(srcs) == 2, "shard not swept -> history silently truncated"


def test_edgar_form_filter_drops_8k():
    d = _edgar()
    d.enumerate_keys({"forms": ["10-K"]})
    forms_in_titles = [s.title.split()[0] for s in d.fetch_for_key("1318605")]
    assert forms_in_titles == ["10-K", "10-K"]  # 8-K excluded


def test_edgar_date_filter():
    d = _edgar()
    d.enumerate_keys({"forms": ["10-K"], "since": "2020"})
    assert len(d.fetch_for_key("1318605")) == 1  # 2011 10-K dropped


def test_edgar_source_fields():
    d = _edgar()
    d.enumerate_keys({"forms": ["10-K"], "since": "2020"})
    s = d.fetch_for_key("1318605")[0]
    # domain canonicalized so corroboration's reliability() prior (keyed on
    # exactly "sec.gov") fires instead of falling back to the 0.5 default
    assert s.domain == "sec.gov"
    # archive URL uses the UN-padded CIK and the dash-stripped accession number
    assert "edgar/data/1318605/000131860523000033/tsla-20221231.htm" in str(s.url)
    assert s.publish_time == datetime(2023, 1, 31)
    assert "R&D expenses" in s.main_text


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov
# --------------------------------------------------------------------------- #
def test_ctgov_param_aliases():
    p = _ctgov()._build_params({"intervention": "pembrolizumab", "phase": 3})
    assert p["query.intr"] == "pembrolizumab"
    assert p["filter.advanced"] == "AREA[Phase]PHASE3"


def test_ctgov_follows_next_page_token():
    """Three trials split across two pages; nextPageToken must be followed."""
    ncts = list(_ctgov().enumerate_keys({"intervention": "pembrolizumab", "phase": 3}))
    assert ncts == ["NCT00000001", "NCT00000002", "NCT00000003"]


def test_ctgov_flattens_study():
    s = _ctgov().fetch_for_key("NCT00000001")[0]
    assert s.domain == "clinicaltrials.gov"
    assert "Enrollment: 123" in s.main_text
    assert "Pembrolizumab" in s.main_text
    assert "PHASE3" in s.main_text


# --------------------------------------------------------------------------- #
# Runner (with injected stubs -> no network, no LLM)
# --------------------------------------------------------------------------- #
def test_runner_persists_and_certifies(tmp_path, monkeypatch):
    import webagg.schema_addressable as sa
    from sqlalchemy import text

    monkeypatch.setattr(sa.config, "RUNS_DIR", tmp_path)

    def fake_relevant(src, query):
        return True

    def fake_extract(src, query):
        return [Mention(
            mention_id=f"{src.source_id}:enrollment:x",
            source_id=src.source_id,
            entity_surface=src.title, record_kind="clinical_trial",
            attribute="enrollment", value="123",
            passage=src.main_text[:50], extracted_at=datetime.utcnow(),
        )]

    result = run_schema_addressable(
        "phase 3 pembrolizumab trials",
        _ctgov(),
        query_filter={"intervention": "pembrolizumab", "phase": 3},
        run_id="test_ct",
        relevance_fn=fake_relevant,
        extract_fn=fake_extract,
    )
    assert result["keys_swept"] == 3
    assert result["mentions_found"] == 3

    session = result["session"]
    n_sources = session.execute(text("SELECT COUNT(*) FROM sources")).scalar()
    n_mentions = session.execute(text("SELECT COUNT(*) FROM mentions")).scalar()
    assert n_sources == 3 and n_mentions == 3

    # the Theorem 3 certificate must be recorded with delta_F = 0
    cert = session.execute(
        text("SELECT extra FROM measurements WHERE metric='schema_complete'")
    ).fetchone()
    assert cert is not None
    extra = cert[0] if isinstance(cert[0], dict) else __import__("json").loads(cert[0])
    assert extra["delta_F"] == 0.0
