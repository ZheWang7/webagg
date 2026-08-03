"""Sec. 15 (withheld-registry oracle) -- the --deny mechanism, offline.

One test per duty (repo convention):
  test_normalize_aliases_and_typo_loud   -> `sec` expands, dotted passes
                                            through, a typo RAISES
  test_blocks_suffix_semantics           -> subdomains blocked, lookalike
                                            domains and ports handled
  test_fetch_layer_refuses_before_network-> denied URL: no GET is attempted
  test_fetch_layer_catches_redirect      -> allowed URL 301ing into a denied
                                            domain is refused post-redirect
  test_pipeline_drops_and_logs_denied    -> the search-result layer drops
                                            the URL pre-fetch, logs
                                            denylist_active + registry_denied,
                                            and the run DB stays clean
  test_assert_no_denied_sources_loud     -> a contaminated run DB raises;
                                            rejected_sources counts too
  test_oracle_sweep_ignores_denylist     -> the schema driver path bypasses
                                            the denylist by construction
  test_per_run_isolation                 -> set_denylist(()) clears; a new
                                            run cannot inherit the last one
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from webagg import fetch, pipeline
from webagg.denylist import (Denylist, assert_no_denied_sources,
                             clear_denylist, get_denylist, normalize,
                             set_denylist)
from webagg.frontier import Formulation
from webagg.schema_addressable import EDGARDriver, run_schema_addressable
from webagg.storage import (MeasurementRow, RejectedSourceRow, SourceRow,
                            get_session)
from webagg.type_defs import Source


# --------------------------------------------------------------------------- #
# module-state hygiene: every test starts and ends with an empty denylist,
# so a failing test can never leak a denial into an unrelated suite.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _clean_denylist():
    clear_denylist()
    yield
    clear_denylist()


def _mk_source(url: str) -> SourceRow:
    return SourceRow(source_id=uuid.uuid4().hex, url=url,
                     domain=url.split("/")[2], fetch_time=datetime.utcnow())


# --------------------------------------------------------------------------- #
# 1. normalize: the CLI grammar
# --------------------------------------------------------------------------- #
def test_normalize_aliases_and_typo_loud():
    # the guide's spelling `--deny sec` -> the EDGAR suffix
    assert normalize(["sec"]) == ("sec.gov",)
    # dotted entries pass through verbatim (experimenter widens the scope)
    assert normalize(["sec", "bamsec.com"]) == ("sec.gov", "bamsec.com")
    # dedupe, order kept; leading dots and case are normalized away
    assert normalize([".SEC.gov", "sec"]) == ("sec.gov",)
    # a typo'd bare token must FAIL the run, not silently deny nothing --
    # a no-op denylist would invalidate the whole withheld-registry claim
    with pytest.raises(ValueError):
        normalize(["sce"])


# --------------------------------------------------------------------------- #
# 2. blocks: suffix semantics
# --------------------------------------------------------------------------- #
def test_blocks_suffix_semantics():
    dl = Denylist(["sec"])
    assert dl.blocks("https://www.sec.gov/Archives/x")       # subdomain
    assert dl.blocks("https://data.sec.gov/submissions/y")   # the API host
    assert dl.blocks("https://SEC.GOV:443/z")                # case + port
    assert not dl.blocks("https://mysec.gov/a")              # different label
    assert not dl.blocks("https://sec.gov.evil.com/b")       # suffix is evil.com
    assert not dl.blocks("not a url")                        # no host -> no block
    assert not Denylist(()).blocks("https://www.sec.gov/")   # empty = allow all


# --------------------------------------------------------------------------- #
# 3. fetch layer: refusal BEFORE the network
# --------------------------------------------------------------------------- #
def test_fetch_layer_refuses_before_network(monkeypatch):
    set_denylist(["sec"])
    fetch.clear_fetch_cache()

    def _explode(url):
        raise AssertionError("network was reached for a denied URL")
    monkeypatch.setattr(fetch, "_get", _explode)

    assert fetch.fetch_url("https://data.sec.gov/x", "f1") is None
    # the denial is on the hit log for the pipeline to persist
    assert get_denylist().hits == [{"url": "https://data.sec.gov/x",
                                    "layer": "fetch"}]


# --------------------------------------------------------------------------- #
# 4. fetch layer: the redirect catch
# --------------------------------------------------------------------------- #
def test_fetch_layer_catches_redirect(monkeypatch):
    set_denylist(["sec"])
    fetch.clear_fetch_cache()

    # an allowed URL whose FINAL location (after follow_redirects) is the
    # registry: fetch_url must refuse on r.url, not the requested url
    fake_resp = SimpleNamespace(status_code=200,
                                headers={"content-type": "text/html"},
                                text="<html>" + "x" * 500 + "</html>",
                                url="https://www.sec.gov/final")
    monkeypatch.setattr(fetch, "_get", lambda url: fake_resp)

    assert fetch.fetch_url("https://sneaky.example.com/go", "f1") is None
    assert get_denylist().hits[-1]["layer"] == "fetch_redirect"
    # and the miss is cached: the same URL is not re-tried this run
    assert fetch._CACHE["https://sneaky.example.com/go"] is None


# --------------------------------------------------------------------------- #
# 5. the pipeline's search-result layer (integration, all seams faked)
# --------------------------------------------------------------------------- #
def test_pipeline_drops_and_logs_denied(monkeypatch, tmp_path):
    run_id = f"t15deny_{uuid.uuid4().hex[:8]}"

    # seams: no Serper key, no LLM -- fakes at exactly the injection points
    class _FakeSearch:
        def search(self, query, k=10, formulation_id=""):
            return [{"url": "https://www.sec.gov/Archives/formd.xml",
                     "title": "", "snippet": "", "formulation_id": formulation_id},
                    {"url": "https://news.example.com/spacex-round",
                     "title": "", "snippet": "", "formulation_id": formulation_id}]
    monkeypatch.setattr(pipeline, "SerperBackend", _FakeSearch)
    monkeypatch.setattr(pipeline, "seed_formulations",
                        lambda q: [Formulation(formulation_id="f1",
                                               query=q)])

    fetched: list[str] = []

    def _fake_fetch(url, formulation_id):
        fetched.append(url)
        return None                     # dead URL: loop continues, no LLM
    monkeypatch.setattr(pipeline, "fetch_url", _fake_fetch)

    state, session = pipeline.run_query(
        "spacex funding rounds", run_id=run_id, max_steps=1,
        deny=("sec",))
    try:
        # the denied result never reached fetch; the allowed one did
        assert fetched == ["https://news.example.com/spacex-round"]
        rows = session.query(MeasurementRow).filter_by(run_id=run_id).all()
        by_metric = {}
        for r in rows:
            by_metric.setdefault(r.metric, []).append(r)
        # the DB self-describes its denial scope ...
        assert by_metric["denylist_active"][0].extra["suffixes"] == ["sec.gov"]
        # ... and records the one unique denied URL, at the result layer
        denied = by_metric["registry_denied"]
        assert len(denied) == 1
        assert denied[0].extra == {"url": "https://www.sec.gov/Archives/formd.xml",
                                   "layer": "search_result"}
        # the closing contamination check passed inside run_query already;
        # re-run it here the way the grading harness will
        assert_no_denied_sources(session, Denylist(["sec"]))
    finally:
        engine = session.get_bind()
        session.close()
        engine.dispose()
        try:
            Path(f"data/runs/{run_id}.sqlite").unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 6. the contamination check is LOUD, and rejections count
# --------------------------------------------------------------------------- #
def test_assert_no_denied_sources_loud(tmp_path):
    session = get_session(str(tmp_path / "t.sqlite"))
    dl = Denylist(["sec"])

    assert_no_denied_sources(session, dl)          # clean DB passes

    session.add(_mk_source("https://ok.example.com/a"))
    session.commit()
    assert_no_denied_sources(session, dl)          # allowed source passes

    # a REJECTED page on a denied domain is still contamination: it was
    # fetched and read by the relevance filter
    session.add(RejectedSourceRow(source_id="r1",
                                  url="https://efts.sec.gov/search"))
    session.commit()
    with pytest.raises(RuntimeError, match="withheld-registry violation"):
        assert_no_denied_sources(session, dl)

    session.close()


# --------------------------------------------------------------------------- #
# 7. the oracle path bypasses the denylist BY CONSTRUCTION
# --------------------------------------------------------------------------- #
def test_oracle_sweep_ignores_denylist(tmp_path):
    # deny sec.gov globally -- exactly the state during an experiment where
    # build_truth is (wrongly) run in the same process as agent runs --
    # then sweep a MockTransport EDGAR: every filing must still arrive,
    # because the driver's own httpx client never consults fetch_url.
    set_denylist(["sec"])

    xml = ("<?xml version='1.0'?><edgarSubmission><primaryIssuer>"
           "<cik>99</cik><entityName>T Corp</entityName></primaryIssuer>"
           "<offeringData><typeOfFiling><newOrAmendment>"
           "<isAmendment>false</isAmendment></newOrAmendment>"
           "<dateOfFirstSale><value>2024-01-10</value></dateOfFirstSale>"
           "</typeOfFiling><offeringSalesAmounts>"
           "<totalOfferingAmount>8000000</totalOfferingAmount>"
           "<totalAmountSold>5000000</totalAmountSold>"
           "</offeringSalesAmounts></offeringData></edgarSubmission>")

    def _handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "submissions" in url:
            return httpx.Response(200, json={
                "name": "T Corp",
                "filings": {"recent": {
                    "form": ["D"], "accessionNumber": ["0000000099-24-000001"],
                    "filingDate": ["2024-01-12"],
                    "primaryDocument": ["xslFormDX01/primary_doc.xml"]},
                    "files": []}})
        return httpx.Response(200, text=xml,
                              headers={"content-type": "application/xml"})

    driver = EDGARDriver(client=httpx.Client(transport=httpx.MockTransport(_handler)),
                         min_interval_s=0.0)
    run_id = f"t15deny_orc_{uuid.uuid4().hex[:8]}"
    got: list[Source] = []
    out = run_schema_addressable(
        "form d rounds", driver, query_filter={"ciks": ["99"], "forms": ["D", "D/A"]},
        run_id=run_id, as_oracle=True,
        extract_fn=lambda src, q: got.append(src) or [])
    try:
        assert len(got) == 1              # the registry read went through
        assert "sec.gov" in str(got[0].url)
    finally:
        engine = out["session"].get_bind()
        out["session"].close()
        engine.dispose()
        try:
            Path(out["db_path"]).unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# 8. per-run isolation of the module slot
# --------------------------------------------------------------------------- #
def test_per_run_isolation():
    set_denylist(["sec"])
    assert get_denylist().blocks("https://www.sec.gov/")
    # the NEXT run installs its own (empty) list -- run_query does this
    # unconditionally, so a previous experiment cannot leak forward
    set_denylist(())
    assert not get_denylist().blocks("https://www.sec.gov/")
