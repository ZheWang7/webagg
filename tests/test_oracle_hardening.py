"""NHL oracle driver: schema traps, self-validation, transport hardening,
and the strict candidate-line parser. All clients are fakes — offline."""
import pytest
from requests.exceptions import Timeout as ReqTimeout

from webaggs.oracle_nhl import NHLDriver, OracleError
from webaggs.truth import parse_candidate_line


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Client:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _landing(season_totals, career_pts, career_gp, first="Benchy",
             last="Marks"):
    return {"firstName": {"default": first}, "lastName": {"default": last},
            "careerTotals": {"regularSeason": {"points": career_pts,
                                               "gamesPlayed": career_gp}},
            "seasonTotals": season_totals}


def _row(season, pts, gp, league="NHL", game_type=2, team="Oilers"):
    e = {"season": season, "leagueAbbrev": league, "gamesPlayed": gp,
         "points": pts, "teamName": {"default": team}}
    if game_type is not None:
        e["gameTypeId"] = game_type
    return e


def _drv(payload):
    return NHLDriver(pace_seconds=0, retry_cooldown=0,
                     client=_Client([_Resp(payload)]))


# --- schema traps ------------------------------------------------------------
def test_filters_leagues_and_playoffs_and_sums_traded_season():
    rows = [
        _row(20152016, 48, 45),                         # NHL regular: in
        _row(20152016, 10, 12, game_type=3),            # NHL playoffs: OUT
        _row(20152016, 30, 20, league="AHL"),           # AHL: OUT
        _row(20142015, 120, 60, league="OHL"),          # junior: OUT
        _row(20162017, 40, 50, team="Oilers"),          # traded season,
        _row(20162017, 25, 30, team="Canucks"),         # two NHL rows: SUMMED
        # v2 leagueAbbrev shape must also be accepted:
        {"season": 20172018, "leagueAbbrev": {"default": "NHL"},
         "gameTypeId": 2, "gamesPlayed": 10, "points": 7,
         "teamName": {"default": "Canucks"}},
    ]
    name, out = _drv(_landing(rows, career_pts=48 + 65 + 7,
                              career_gp=45 + 80 + 10)).career_regular_season(1)
    assert name == "Benchy Marks"
    assert [(r["season"], r["pts"], r["gp"]) for r in out] == [
        ("2015-16", 48, 45), ("2016-17", 65, 80), ("2017-18", 7, 10)]
    assert out[1]["team"] == "Oilers/Canucks"           # traded: both teams


def test_missing_game_type_on_nhl_row_raises_never_guesses():
    rows = [_row(20152016, 48, 45, game_type=None)]
    with pytest.raises(OracleError, match="gameTypeId"):
        _drv(_landing(rows, 48, 45)).career_regular_season(1)


def test_unknown_league_shape_raises():
    rows = [{"season": 20152016, "leagueAbbrev": 7, "gameTypeId": 2,
             "gamesPlayed": 1, "points": 1, "teamName": {"default": "X"}}]
    with pytest.raises(OracleError, match="leagueAbbrev"):
        _drv(_landing(rows, 1, 1)).career_regular_season(1)


def test_self_validation_against_career_totals():
    """THE anti-delusion guard: filtered rows must reproduce the API's own
    careerTotals, or no truth is emitted."""
    rows = [_row(20152016, 48, 45)]
    with pytest.raises(OracleError, match="SELF-VALIDATION"):
        _drv(_landing(rows, career_pts=50, career_gp=45)).career_regular_season(1)
    with pytest.raises(OracleError, match="self-validate"):
        payload = _landing(rows, 48, 45)
        del payload["careerTotals"]                     # totals absent: refuse
        _drv(payload).career_regular_season(1)


def test_zero_nhl_rows_and_bad_season_raise():
    with pytest.raises(OracleError, match="zero NHL"):
        _drv(_landing([_row(20152016, 9, 9, league="AHL")],
                      0, 0)).career_regular_season(1)
    with pytest.raises(OracleError, match="season identifier"):
        _drv(_landing([_row(20152017, 9, 9)], 9, 9)).career_regular_season(1)


# --- transport ----------------------------------------------------------------
def test_timeout_is_retried_once_then_succeeds():
    ok = _Resp(_landing([_row(20152016, 48, 45)], 48, 45))
    client = _Client([ReqTimeout("hang"), ok])
    d = NHLDriver(pace_seconds=0, retry_cooldown=0, client=client)
    name, rows = d.career_regular_season(1)
    assert client.calls == 2 and rows[0]["pts"] == 48


def test_timeout_twice_raises_loudly():
    client = _Client([ReqTimeout("hang"), ReqTimeout("hang")])
    d = NHLDriver(pace_seconds=0, retry_cooldown=0, client=client)
    with pytest.raises(OracleError, match="Timeout"):
        d.career_regular_season(1)


def test_404_is_not_retried_and_names_the_remedy():
    client = _Client([_Resp({}, status=404)])
    d = NHLDriver(pace_seconds=0, retry_cooldown=0, client=client)
    with pytest.raises(OracleError, match="nhl.com page URL"):
        d.career_regular_season(999)
    assert client.calls == 1                            # retrying 404: pointless


# --- candidate parser ----------------------------------------------------------
def test_parse_candidate_line():
    assert parse_candidate_line("Connor McDavid | 8478402") == \
        ("Connor McDavid", 8478402)
    assert parse_candidate_line("Connor McDavid") == ("Connor McDavid", None)
    for bad in ("| 123", "Name |", "Name | 12x3"):
        with pytest.raises(ValueError):
            parse_candidate_line(bad)


# --- candidate resolver helpers (dev tooling, but its matching must be right)
def test_resolver_name_matching_and_result_parsing():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from resolve_candidates import _result_fields, names_match

    assert names_match("Josh Ho-Sang", "Joshua Ho-Sang")      # nickname fold
    assert names_match("Fabian Brunnstrom", "Fabian Brunnström")
    assert names_match("JERRY D'AMIGO", "Jerry D'Amigo")
    assert not names_match("Sebastian Aho", "Connor McDavid")
    # same surname + initial IS folded by the ER rule — the reason the
    # resolver demands a UNIQUE id among matches, never picks a best one
    assert names_match("Jack Smith", "John Smith")

    assert _result_fields({"playerId": "8478402",
                           "name": "Connor McDavid"}) == (8478402,
                                                          "Connor McDavid")
    assert _result_fields({"id": 1, "firstName": {"default": "A"},
                           "lastName": {"default": "B"}}) == (1, "A B")
    assert _result_fields({"weird": 1}) is None
    assert _result_fields({"playerId": "abc", "name": "X"}) is None
