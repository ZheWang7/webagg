"""Regressions for the stats.nba.com hardening (found by Zhe's preflight,
2026-10-20): a read-timeout must be RETRIED, not raised through, and the
'Name | PLAYER_ID' bypass must parse strictly."""
import pytest
from requests.exceptions import Timeout as ReqTimeout

from webaggs.oracle_nba import NBAStatsDriver, OracleError
from webaggs.truth import parse_candidate_line


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload

    def json(self):
        return self._p


class _FlakyClient:
    """Hangs (raises Timeout) n times, then answers."""

    def __init__(self, fail_times, payload):
        self.fail_times = fail_times
        self.payload = payload
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None,
            proxies=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ReqTimeout("read timed out")
        return _Resp(self.payload)


_PAYLOAD = {"resultSets": [{"name": "SeasonTotalsRegularSeason",
                            "headers": ["SEASON_ID", "TEAM_ABBREVIATION",
                                        "PTS", "GP"],
                            "rowSet": [["2018-19", "DAL", 668, 73]]}]}


def test_timeout_is_retried_once_then_succeeds():
    client = _FlakyClient(1, _PAYLOAD)
    d = NBAStatsDriver(pace_seconds=0, client=client, retry_cooldown=0)
    rows = d.career_regular_season(1628973)
    assert client.calls == 2 and rows[0]["pts"] == 668


def test_timeout_twice_raises_with_the_bypass_remedy():
    d = NBAStatsDriver(pace_seconds=0, client=_FlakyClient(99, _PAYLOAD),
                       retry_cooldown=0)
    with pytest.raises(OracleError, match="Name \\| PLAYER_ID"):
        d.career_regular_season(1628973)


def test_parse_candidate_line():
    assert parse_candidate_line("Jalen Brunson") == ("Jalen Brunson", None)
    assert parse_candidate_line("Jalen Brunson | 1628973") == \
        ("Jalen Brunson", 1628973)
    for bad in ("| 123", "Name |", "Name | 12x3"):
        with pytest.raises(ValueError):
            parse_candidate_line(bad)


class _ProxySpy:
    def __init__(self):
        self.proxies = "unset"

    def get(self, url, params=None, headers=None, timeout=None, proxies=None):
        self.proxies = proxies
        return _Resp(_PAYLOAD)


def test_oracle_proxy_env_routes_oracle_calls(monkeypatch):
    spy = _ProxySpy()
    d = NBAStatsDriver(pace_seconds=0, client=spy, retry_cooldown=0)
    monkeypatch.delenv("WEBAGG_ORACLE_PROXY", raising=False)
    d.career_regular_season(1)
    assert spy.proxies is None                     # unset: direct
    monkeypatch.setenv("WEBAGG_ORACLE_PROXY", "http://us-proxy:8080")
    d.career_regular_season(1)
    assert spy.proxies == {"http": "http://us-proxy:8080",
                           "https": "http://us-proxy:8080"}
