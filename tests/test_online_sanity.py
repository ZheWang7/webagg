"""ONLINE sanity preflight — the only tests that touch the real network.
Opt-in: RUN_ONLINE=1. The oracle checks exercise the REAL driver against
OpenAlex, including a full cursor walk with self-validation on a small
early-cutoff slice (papers citing 'Attention Is All You Need' published
before 2017-09-01 — a stable, finished set)."""
import os

import pytest

from webaggs import config
from webaggs.fetch import fetch_url

pytestmark = [
    pytest.mark.online,
    pytest.mark.skipif(os.environ.get("RUN_ONLINE") != "1",
                       reason="online sanity is opt-in: set RUN_ONLINE=1"),
]

ATTENTION_WID = "W2626778328"       # openalex.org/W2626778328


def test_oracle_work_meta_reachable_and_sane():
    from webaggs.oracle_openalex import OpenAlexDriver
    m = OpenAlexDriver().work_meta(ATTENTION_WID)
    assert "attention is all you need" in m["title"].lower()
    assert m["year"] == 2017
    assert m["cited_by_count"] > 50000            # grows; floor is stable


def test_oracle_citers_walk_self_validates_on_early_slice():
    from webaggs.oracle_openalex import OpenAlexDriver
    rows = OpenAlexDriver().citers(ATTENTION_WID, "2017-09-01")
    # a finished historical slice: non-empty, modest, fully keyed
    assert 0 < len(rows) < 2000
    assert all(r["key"].startswith(("doi:", "t:")) for r in rows)
    assert len({r["key"] for r in rows}) == len(rows)


def test_open_web_fetch_classifies_ok():
    src = fetch_url("https://www.wikipedia.org", "sanity")
    assert src.status == "ok" and len(src.main_text) > 500


def test_deny_list_blocks_the_real_oracle():
    src = fetch_url("https://api.openalex.org/works", "sanity")
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
