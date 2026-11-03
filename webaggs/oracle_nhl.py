"""The withheld oracle: the NHL's official stats API (webaggr.pdf §9 role).

Endpoint: GET https://api-web.nhle.com/v1/player/{id}/landing — the league's
own public, unauthenticated API. Truth is DEFINED by this API and DENIED to
the agent at run time (config.DENY_DOMAINS covers nhl.com/nhle.com). This
driver is used ONLY by the truth-build script, never by the pipeline.

Response shape (verified 2026-11-03 against the maintained nhl-api-py
wrapper's response docs, not from memory):
  firstName.default / lastName.default        — official display name
  careerTotals.regularSeason.{points,gamesPlayed}
  seasonTotals: list, ONE ENTRY PER (season, league, game type, team):
    season       int YYYYYYYY (e.g. 20152016)
    gameTypeId   int: 2 = regular season, 3 = playoffs
    leagueAbbrev "NHL"/"AHL"/"OHL"/... (string, or {"default": ...}; both
                 accepted, anything else raises)
    teamName.default, gamesPlayed, points
    sequence     order within a season for players traded mid-season

The traps this encodes, each a loud failure, never a guess:
  * playoff rows (gameTypeId 3) and non-NHL leagues share the list — filter
    on BOTH; an NHL entry MISSING gameTypeId raises (filtering by absence
    would silently admit playoff points);
  * a traded season is several rows (per team) — summed into ONE
    (player, season) record per the pre-registered scope rule;
  * SELF-VALIDATION: the filtered rows' summed points and games MUST equal
    the API's own careerTotals.regularSeason figures, or the driver refuses
    to emit truth. A wrong truth table is structurally impossible as long
    as the API agrees with itself.

No name-search endpoint is implemented: none is verifiably documented, so
candidates.txt carries explicit ids ('Name | 8478402') taken from the
player's nhl.com page URL (e.g. nhl.com/player/connor-mcdavid-8478402).
"""
from __future__ import annotations

import time
from collections import defaultdict

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout

_BASE = "https://api-web.nhle.com/v1"
_UA = {"User-Agent": "webagg-sports research client (Exp 2-S truth build)",
       "Accept": "application/json"}


class OracleError(RuntimeError):
    pass


def _league_of(entry: dict) -> str:
    """leagueAbbrev is a string (v1) or {'default': str} (v2). Anything else
    is an unknown shape: raise, never coerce."""
    v = entry.get("leagueAbbrev")
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and isinstance(v.get("default"), str):
        return v["default"]
    raise OracleError(f"Unrecognized leagueAbbrev shape in seasonTotals "
                      f"entry: {v!r} — the API schema changed; refusing to "
                      "guess which league this row belongs to.")


def _season_str(season_int) -> str:
    """20152016 -> '2015-16', validated strictly."""
    s = str(season_int)
    if len(s) != 8 or not s.isdigit() or int(s[4:]) != int(s[:4]) + 1:
        raise OracleError(f"Unparseable season identifier {season_int!r} "
                          "(expected YYYYYYYY with consecutive years).")
    return f"{s[:4]}-{s[6:]}"


class NHLDriver:
    """Oracle driver for build_cohort.py. Paced, retried once on transient
    errors, loud on everything else (the EDGAR lesson: a blocked fetch must
    never look like data)."""

    def __init__(self, pace_seconds: float = 1.0, client=None,
                 retry_cooldown: float = 10.0):
        self.pace = pace_seconds
        self.client = client or requests     # tests inject fakes
        self.retry_cooldown = retry_cooldown
        self._last = 0.0

    def _get(self, resource: str) -> dict:
        wait = self.pace - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)                 # pace EVERY request
        detail = "no response"
        for attempt in (1, 2):
            self._last = time.time()
            try:
                resp = self.client.get(f"{_BASE}/{resource}", headers=_UA,
                                       timeout=(10, 30))
            except (ReqTimeout, ReqConnectionError) as e:
                detail = type(e).__name__    # transient: one paced retry
                if attempt == 1:
                    time.sleep(self.retry_cooldown)
                continue
            if resp.status_code == 200:
                return resp.json()
            detail = f"HTTP {resp.status_code}"   # 404 = bad player id
            if resp.status_code == 404:
                break                        # retrying a 404 is pointless
            if attempt == 1:
                time.sleep(self.retry_cooldown)
        raise OracleError(f"Oracle request failed: {resource} ({detail}). "
                          "Raising loudly — an empty truth table must never "
                          "look like data. A 404 means a wrong player id: "
                          "take the id from the player's nhl.com page URL.")

    def career_regular_season(self, player_id: int) -> tuple[str, list[dict]]:
        """-> (official display name, [{season:'YYYY-YY', team, pts, gp}]).

        One record per (player, season): NHL regular-season rows only,
        traded seasons summed across teams, self-validated against the
        API's own careerTotals before anything is returned.
        """
        data = self._get(f"player/{player_id}/landing")

        try:
            name = (f"{data['firstName']['default']} "
                    f"{data['lastName']['default']}")
        except (KeyError, TypeError) as e:
            raise OracleError(f"Player {player_id}: name fields missing from "
                              f"landing payload ({e!r}) — schema changed?")

        entries = data.get("seasonTotals")
        if not entries:
            raise OracleError(f"Player {player_id} has no seasonTotals — "
                              "wrong id or schema change; raising loudly.")

        by_season: dict[str, dict] = defaultdict(
            lambda: {"pts": 0, "gp": 0, "teams": []})
        for e in entries:
            if _league_of(e) != "NHL":
                continue                     # junior/minor/intl leagues: out
            if "gameTypeId" not in e:
                # filtering playoffs by ABSENCE of the field would silently
                # admit playoff points if the schema drifts — refuse instead
                raise OracleError(
                    f"NHL seasonTotals entry without gameTypeId for player "
                    f"{player_id} (season {e.get('season')!r}) — cannot "
                    "distinguish regular season from playoffs; refusing.")
            if e["gameTypeId"] != 2:
                continue                     # playoffs (3), preseason (1): out
            season = _season_str(e["season"])
            rec = by_season[season]
            rec["pts"] += int(e.get("points") or 0)
            rec["gp"] += int(e.get("gamesPlayed") or 0)
            team = (e.get("teamName") or {}).get("default", "?")
            rec["teams"].append(team)        # traded season: several rows

        if not by_season:
            raise OracleError(f"Player {player_id} ({name}) has zero NHL "
                              "regular-season rows — not an NHL player, or "
                              "a schema change; raising loudly.")

        # --- self-validation against the API's own career totals ---------
        career = (data.get("careerTotals") or {}).get("regularSeason") or {}
        if "points" not in career or "gamesPlayed" not in career:
            raise OracleError(f"Player {player_id}: careerTotals.regular"
                              "Season missing — cannot self-validate; "
                              "refusing to emit unvalidated truth.")
        sum_pts = sum(r["pts"] for r in by_season.values())
        sum_gp = sum(r["gp"] for r in by_season.values())
        if sum_pts != int(career["points"]) or sum_gp != int(career["gamesPlayed"]):
            raise OracleError(
                f"SELF-VALIDATION FAILED for {name} ({player_id}): filtered "
                f"seasonTotals sum to {sum_pts} pts / {sum_gp} gp but the "
                f"API's careerTotals.regularSeason says "
                f"{career['points']} pts / {career['gamesPlayed']} gp. The "
                "filter or the schema is wrong — refusing to write truth.")

        rows = [{"season": season, "team": "/".join(rec["teams"]),
                 "pts": rec["pts"], "gp": rec["gp"]}
                for season, rec in sorted(by_season.items())]
        return name, rows
