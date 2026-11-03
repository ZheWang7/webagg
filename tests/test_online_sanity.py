"""ONLINE sanity preflight — the only tests that touch the real network.

Opt-in by design: skipped unless RUN_ONLINE=1, so `pytest` stays fully
offline. Run once from the machine that will do the live runs, BEFORE any
spend:

    $env:RUN_ONLINE="1"; pytest -m online -v        (PowerShell)

The two key-dependent checks cost ~$0.001 (Serper) and <$0.01 (OpenAI) and
additionally skip when their key is absent. Nothing asserts numbers that
can go stale — only structural invariants plus one immutable fact
(McDavid's rookie season), and the driver ALREADY self-validates every
career against the API's own careerTotals before returning.
"""
import os

import pytest

from webaggs import config
from webaggs.fetch import fetch_url
from webaggs.truth import season_key

pytestmark = [
    pytest.mark.online,
    pytest.mark.skipif(os.environ.get("RUN_ONLINE") != "1",
                       reason="online sanity is opt-in: set RUN_ONLINE=1"),
]


MCDAVID_ID = 8478402        # from nhl.com/player/connor-mcdavid-8478402


def test_oracle_reachable_and_self_validates():
    """The landing endpoint — the only one the truth build uses — answers,
    the driver's league/game-type filter reproduces the API's own career
    totals (its built-in self-validation), and the rows have the shape the
    truth builder expects."""
    from webaggs.oracle_nhl import NHLDriver
    name, rows = NHLDriver().career_regular_season(MCDAVID_ID)
    assert name == "Connor McDavid"
    seasons = [r["season"] for r in rows]
    assert "2015-16" in seasons                 # rookie year: immutable fact
    assert seasons == sorted(seasons) and len(seasons) == len(set(seasons))
    for r in rows:
        season_key(r["season"])                 # parses, or raises loudly
        assert isinstance(r["pts"], int) and 0 <= r["pts"] <= config.POINTS_MAX
        assert r["gp"] >= 0


def test_open_web_fetch_classifies_ok():
    """A stable public page comes back status='ok' with real text — the
    fetcher, UA and outbound network all work from this machine."""
    src = fetch_url("https://www.wikipedia.org", "sanity")
    assert src.status == "ok" and len(src.main_text) > 500


def test_deny_list_blocks_the_real_oracle():
    """The contrast line: oracle URLs are refused BEFORE any request."""
    for url in ("https://api-web.nhle.com/v1/player/8478402/landing",
                "https://www.nhl.com/player/connor-mcdavid-8478402"):
        src = fetch_url(url, "sanity")
        assert src.status == "denied" and src.main_text == ""


@pytest.mark.skipif(not os.environ.get(config.SERPER_API_KEY_ENV),
                    reason="SERPER_API_KEY not set")
def test_serper_key_works():
    from webaggs.search import SerperBackend
    results = SerperBackend().search("NHL", k=3)
    assert results and all(r["url"].startswith("http") for r in results)


@pytest.mark.skipif(not os.environ.get(config.OPENAI_API_KEY_ENV),
                    reason="OPENAI_API_KEY not set")
def test_llm_key_and_json_mode_work():
    from webaggs.llm import call_llm
    out = call_llm('Reply with JSON exactly: {"ok": true}', "go")
    assert out.get("ok") is True
