"""ONLINE sanity preflight — the only tests that touch the real network.
Opt-in: RUN_ONLINE=1. They exercise the REAL oracle driver (OpenCitations,
including its self-validation) and one REAL page of each enumerator
(OpenAlex, Europe PMC) — a dry run of exactly what truth build and the
live walk do, at $0."""
import os

import pytest

from webaggs import config
from webaggs.fetch import fetch_url

pytestmark = [
    pytest.mark.online,
    pytest.mark.skipif(os.environ.get("RUN_ONLINE") != "1",
                       reason="online sanity is opt-in: set RUN_ONLINE=1"),
]

# AlphaFold 2 — a biomedical giant known to every index
ALPHAFOLD = dict(title="Highly accurate protein structure prediction with "
                       "AlphaFold", doi="10.1038/s41586-021-03819-2",
                 pmid="34265844")
# a small, old paper for the oracle walk (OpenCitations' own doc example)
SMALL_DOI = "10.1108/jd-12-2013-0166"


def test_oracle_list_self_validates_against_its_own_count():
    from webaggs.oracle_opencitations import OpenCitationsDriver
    rows, rep = OpenCitationsDriver().citers(SMALL_DOI, "2022-12-31")
    # self-validation ran inside citers(); a finished historical slice:
    assert 0 < len(rows) <= rep["raw_rows"]
    assert all(r["key"].startswith("doi:") for r in rows)
    assert len({r["key"] for r in rows}) == len(rows)


def _first_page(ix):
    from webaggs.indexes import Target
    t = Target(**ALPHAFOLD)
    handle = ix.resolve(t)
    src, page = ix.fetch_page(t, handle, ix.first_cursor, 25, "2023-12-31",
                              "sanity")
    return src, page


def test_openalex_index_serves_a_bounded_keyed_page():
    from webaggs.indexes import OpenAlexIndex
    src, page = _first_page(OpenAlexIndex())
    assert src.status == "ok", src.status
    assert 0 < len(page.rows) <= 25 and page.next_cursor
    assert page.total and page.total > 1000
    assert all(r["doi"] for r in page.rows)        # has_doi:true filter
    assert "api_key" not in src.url                # ledger URL is keyless


def test_europepmc_index_serves_a_bounded_keyed_page():
    from webaggs.indexes import EuropePMCIndex
    src, page = _first_page(EuropePMCIndex())
    assert src.status == "ok", src.status
    assert 0 < len(page.rows) <= 25 and page.next_cursor
    assert page.total and page.total > 1000
    with_doi = sum(1 for r in page.rows if r["doi"])
    assert with_doi >= len(page.rows) // 2         # the DOI universe is there


def test_open_web_fetch_classifies_ok():
    src = fetch_url("https://www.wikipedia.org", "sanity")
    assert src.status == "ok" and len(src.main_text) > 500


def test_deny_list_blocks_the_real_oracle():
    src = fetch_url("https://api.opencitations.net/index/v2/citations/"
                    "doi:10.1108/jd-12-2013-0166", "sanity")
    assert src.status == "denied" and src.main_text == ""


@pytest.mark.skipif(not os.environ.get(config.SERPER_API_KEY_ENV),
                    reason="SERPER_API_KEY not set")
def test_serper_key_works():
    from webaggs.search import SerperBackend
    results = SerperBackend().search("citations", k=3)
    assert results and all(r["url"].startswith("http") for r in results)


@pytest.mark.skipif(not os.environ.get(config.OPENAI_API_KEY_ENV),
                    reason="OPENAI_API_KEY not set")
def test_llm_key_and_json_mode_work():
    from webaggs.llm import call_llm
    out = call_llm('Reply with JSON exactly: {"ok": true}', "go")
    assert out.get("ok") is True
