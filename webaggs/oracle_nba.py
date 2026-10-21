"""The withheld oracle: official NBA stats (webaggr.pdf §9 analogue of EDGAR).

Truth is DEFINED by stats.nba.com and DENIED to the agent at run time
(config.DENY_DOMAINS + fetch.is_denied). This driver is used ONLY by the
truth-build script, never by the pipeline.

Operational care (the EDGAR rate-limit lesson: blocked fetches that return
empty lists look like data): every request is paced, retried once with a
cooldown, and a non-200 or empty payload RAISES instead of returning [].
"""
from __future__ import annotations

import time

import requests

_HEADERS = {
    # stats.nba.com rejects requests without browser-like headers
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
_BASE = "https://stats.nba.com/stats"


class OracleError(RuntimeError):
    pass


class NBAStatsDriver:
    """Schema-addressable driver shape (webaggr.pdf §5.1): enumerate players,
    fetch per-key season rows — the sweep is deterministic over its keys."""

    def __init__(self, pace_seconds: float = 1.5, client=None):
        self.pace = pace_seconds
        self.client = client or requests
        self._last = 0.0

    def _get(self, endpoint: str, params: dict) -> dict:
        wait = self.pace - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)                       # pace EVERY request
        for attempt in (1, 2):
            self._last = time.time()
            resp = self.client.get(f"{_BASE}/{endpoint}", params=params,
                                   headers=_HEADERS, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("resultSets"):
                    return data
            if attempt == 1:
                time.sleep(10)                     # cooldown, then one retry
        raise OracleError(f"Oracle request failed twice: {endpoint} {params} "
                          f"(HTTP {resp.status_code}). Raising loudly — an "
                          "empty truth table must never look like data.")

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
