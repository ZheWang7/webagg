"""The withheld oracle: official NBA stats (webaggr.pdf §9 analogue of EDGAR).

Truth is DEFINED by stats.nba.com and DENIED to the agent at run time
(config.DENY_DOMAINS + fetch.is_denied). This driver is used ONLY by the
truth-build script, never by the pipeline.

Operational care (the EDGAR rate-limit lesson: blocked fetches that return
empty lists look like data): every request is paced, retried once with a
cooldown, and a non-200 or empty payload RAISES instead of returning [].
"""
from __future__ import annotations

import os
import time

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout

# The stats edge hangs non-browser TLS fingerprints (python-requests' JA3)
# regardless of IP — observed identically from Canadian residential, a US
# VPN exit, Azure CI runners and a cloud fetcher. curl_cffi impersonates a
# real Chrome handshake, which is the standard way past that wall.
try:
    from curl_cffi import requests as _crequests
except ImportError:                                   # pragma: no cover
    _crequests = None

_HEADERS = {
    # stats.nba.com rejects requests without browser-like headers
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    # stats.nba.com's bot-wall HANGS incomplete browser fingerprints rather
    # than refusing them (symptom: connect ok, read times out) — these four
    # are the ones its heuristics check beyond the UA/referer set above
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}
_BASE = "https://stats.nba.com/stats"


class OracleError(RuntimeError):
    pass


def _is_transport_error(e: Exception) -> bool:
    """Timeouts/connection failures from requests OR curl_cffi are retryable;
    anything else (a KeyError, a JSON bug) must raise, never be swallowed."""
    if isinstance(e, (ReqTimeout, ReqConnectionError)):
        return True
    mod = type(e).__module__ or ""
    name = type(e).__name__
    return mod.startswith("curl_cffi") and name in (
        "CurlError", "Timeout", "ConnectionError", "RequestsError",
        "RequestException")


class NBAStatsDriver:
    """Schema-addressable driver shape (webaggr.pdf §5.1): enumerate players,
    fetch per-key season rows — the sweep is deterministic over its keys."""

    def __init__(self, pace_seconds: float = 1.5, client=None,
                 retry_cooldown: float = 10.0):
        self.pace = pace_seconds
        if client is not None:
            self.client = client                      # tests inject fakes
        elif _crequests is not None:
            # Chrome-impersonated TLS (curl_cffi); plain requests only as a
            # last resort when curl_cffi is unavailable on the platform
            self.client = _crequests.Session(impersonate="chrome")
        else:                                          # pragma: no cover
            print("[warn] curl_cffi not installed — falling back to plain "
                  "requests, whose TLS fingerprint the stats edge may hang. "
                  "pip install curl_cffi")
            self.client = requests
        self.retry_cooldown = retry_cooldown
        self._last = 0.0

    def _get(self, endpoint: str, params: dict) -> dict:
        wait = self.pace - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)                       # pace EVERY request
        # stats.nba.com blackholes many non-US networks at the edge (TLS
        # connects, reads hang — browsers included). WEBAGG_ORACLE_PROXY
        # routes ONLY these oracle calls through a US exit; Serper/OpenAI
        # traffic is unaffected. (A system-level VPN needs no setting.)
        proxy = os.environ.get("WEBAGG_ORACLE_PROXY")
        proxies = {"http": proxy, "https": proxy} if proxy else None
        detail = "no response"
        for attempt in (1, 2):
            self._last = time.time()
            try:
                # (connect, read): the bot-wall often streams slowly before
                # answering — give reads 60 s, and treat a hang as retryable
                kwargs = {"params": params, "headers": _HEADERS,
                          "timeout": (10, 60)}
                if proxies:
                    kwargs["proxies"] = proxies
                resp = self.client.get(f"{_BASE}/{endpoint}", **kwargs)
            except Exception as e:                # requests AND curl_cffi
                if not _is_transport_error(e):      # real bugs still raise
                    raise
                detail = f"{type(e).__name__}"      # hang/reset: retryable
                if attempt == 1:
                    time.sleep(self.retry_cooldown)
                continue
            if resp.status_code == 200:
                data = resp.json()
                if data.get("resultSets"):
                    return data
            detail = f"HTTP {resp.status_code}"
            if attempt == 1:
                time.sleep(self.retry_cooldown)     # cooldown, then one retry
        raise OracleError(
            f"Oracle request failed twice: {endpoint} {params} ({detail}). "
            "Raising loudly — an empty truth table must never look like "
            "data. If this is the commonallplayers lookup hanging (a known "
            "stats.nba.com bot-wall behavior), bypass it with explicit ids "
            "in cohort/candidates.txt: 'Name | PLAYER_ID'.")

    def find_player_id(self, name: str) -> tuple[int, str]:
        """Resolve a player name to (official id, official display name)."""
        data = self._get("commonallplayers",
                         {"LeagueID": "00", "Season": "2025-26",
                          "IsOnlyCurrentSeason": "0"})
        rs = data["resultSets"][0]
        cols = rs["headers"]
        i_id, i_name = cols.index("PERSON_ID"), cols.index("DISPLAY_FIRST_LAST")
        from .canon import normalize_name
        target = normalize_name(name)
        matches = [(row[i_id], row[i_name]) for row in rs["rowSet"]
                   if normalize_name(row[i_name]) == target]
        if len(matches) != 1:
            raise OracleError(f"Player lookup for {name!r} returned "
                              f"{len(matches)} matches: {matches[:5]} — resolve "
                              "the ambiguity in the candidates file (e.g. the "
                              "two Jalen Johnsons), do not guess.")
        return matches[0]

    def career_regular_season(self, player_id: int) -> list[dict]:
        """One row per (season): league-wide regular-season totals.

        Scope rules (PREREGISTRATION_EXP2S.md): regular season only; a traded
        season is ONE record — the API's 'TOT' row when present, else the
        single team row. G-League/international never appear here, matching
        the query scope 'NBA career points'.
        """
        data = self._get("playercareerstats",
                         {"PlayerID": player_id, "PerMode": "Totals",
                          "LeagueID": "00"})
        rs = next(r for r in data["resultSets"]
                  if r["name"] == "SeasonTotalsRegularSeason")
        cols = rs["headers"]
        i_season = cols.index("SEASON_ID")
        i_team = cols.index("TEAM_ABBREVIATION")
        i_pts, i_gp = cols.index("PTS"), cols.index("GP")
        by_season: dict[str, dict] = {}
        for row in rs["rowSet"]:
            season, team = row[i_season], row[i_team]
            rec = {"season": season, "team": team,
                   "pts": int(row[i_pts] or 0), "gp": int(row[i_gp] or 0)}
            prev = by_season.get(season)
            # 'TOT' (league-wide) supersedes per-team splits for traded seasons
            if prev is None or team == "TOT":
                if prev is None or prev["team"] != "TOT":
                    by_season[season] = rec
        rows = sorted(by_season.values(), key=lambda r: r["season"])
        if not rows:
            raise OracleError(f"Player {player_id} has zero regular-season "
                              "rows — wrong id or oracle block; raising loudly.")
        return rows
