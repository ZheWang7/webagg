"""ONLINE sanity preflight — the only tests that touch the real network.

Opt-in by design: skipped unless RUN_ONLINE=1, so `pytest` stays fully
offline. Run this once from the machine that will do the live runs, BEFORE
any spend, to confirm every external dependency is reachable from there:

    $env:RUN_ONLINE="1"; pytest -m online -v        (PowerShell)

The two key-dependent checks cost ~$0.001 (Serper) and <$0.01 (OpenAI) and
additionally skip when their key is absent. Nothing here asserts exact
historical numbers — only structural invariants that cannot go stale.
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


BRUNSON_ID = 1628973                            # official stats.nba.com id


def test_oracle_career_endpoint_reachable_and_sane():
    """playercareerstats — the endpoint build_cohort actually hammers, one
    call per candidate — answers, and its rows have the shape the truth
    builder expects. This is the hard oracle requirement."""
    from webaggs.oracle_nba import NBAStatsDriver
    rows = NBAStatsDriver().career_regular_season(BRUNSON_ID)
    seasons = [r["season"] for r in rows]
    assert "2018-19" in seasons                 # rookie year: a stable fact
    assert seasons == sorted(seasons) and len(seasons) == len(set(seasons))
    for r in rows:
        season_key(r["season"])                 # parses, or raises loudly
        assert isinstance(r["pts"], int) and 0 <= r["pts"] <= 60000
        assert r["gp"] >= 0


def test_oracle_name_lookup_soft():
    """commonallplayers — the all-history index behind a bot-wall that is
    KNOWN to hang some networks. A hang here is survivable (candidates.txt
    takes 'Name | PLAYER_ID'), so it skips with the remedy instead of
    failing the preflight; any other outcome must still be correct."""
    from webaggs.oracle_nba import NBAStatsDriver, OracleError
    try:
        pid, official = NBAStatsDriver().find_player_id("Jalen Brunson")
    except OracleError as e:
        pytest.skip(f"name-lookup endpoint unavailable from this network "
                    f"({e}); use 'Name | PLAYER_ID' in candidates.txt")
    assert (pid, official) == (BRUNSON_ID, "Jalen Brunson")


def test_open_web_fetch_classifies_ok():
    """A stable public page comes back status='ok' with real text — the
    fetcher, UA and outbound network all work from this machine."""
    src = fetch_url("https://www.wikipedia.org", "sanity")
    assert src.status == "ok" and len(src.main_text) > 500


def test_deny_list_blocks_the_real_oracle():
    """The contrast line: the oracle URL is refused BEFORE any request."""
    src = fetch_url("https://stats.nba.com/stats/playercareerstats", "sanity")
    assert src.status == "denied" and src.main_text == ""


@pytest.mark.skipif(not os.environ.get(config.SERPER_API_KEY_ENV),
                    reason="SERPER_API_KEY not set")
def test_serper_key_works():
    from webaggs.search import SerperBackend
    results = SerperBackend().search("NBA", k=3)
    assert results and all(r["url"].startswith("http") for r in results)


@pytest.mark.skipif(not os.environ.get(config.OPENAI_API_KEY_ENV),
                    reason="OPENAI_API_KEY not set")
def test_llm_key_and_json_mode_work():
    from webaggs.llm import call_llm
    out = call_llm('Reply with JSON exactly: {"ok": true}', "go")
    assert out.get("ok") is True
