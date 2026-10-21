import pytest

from webaggs.oracle_nba import NBAStatsDriver, OracleError
from webaggs.truth import grade, load_truth, write_truth


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Client:
    """Fake HTTP client for the oracle driver (tests only)."""

    def __init__(self, payloads):
        self.payloads = list(payloads)

    def get(self, url, params=None, headers=None, timeout=None,
            proxies=None):
        return self.payloads.pop(0)


def _career_payload():
    return {"resultSets": [{
        "name": "SeasonTotalsRegularSeason",
        "headers": ["SEASON_ID", "TEAM_ABBREVIATION", "PTS", "GP"],
        "rowSet": [
            ["2019-20", "DAL", 450, 60],
            ["2020-21", "DAL", 300, 40],      # traded mid-season:
            ["2020-21", "NYK", 312, 30],      # two team splits...
            ["2020-21", "TOT", 612, 70],      # ...and the league-wide TOT row
        ]}]}


def test_tot_row_supersedes_team_splits():
    """Scope rule: a traded season is ONE record, the league-wide total."""
    driver = NBAStatsDriver(pace_seconds=0, client=_Client([
        _Resp(200, _career_payload())]))
    rows = driver.career_regular_season(1)
    assert [(r["season"], r["pts"]) for r in rows] == \
        [("2019-20", 450), ("2020-21", 612)]


def test_oracle_raises_loudly_on_block():
    """The EDGAR rate-limit lesson: a blocked fetch must RAISE, never return
    an empty list that looks like data."""
    driver = NBAStatsDriver(pace_seconds=0, client=_Client([
        _Resp(403, {}), _Resp(403, {})]))
    with pytest.raises(OracleError):
        driver.career_regular_season(1)


def test_truth_freeze_detects_tampering(tmp_path):
    rows = [{"player": "Testo Sampleson", "player_id": 1,
             "season": "2019-20", "pts": 450, "gp": 60}]
    csv_path = write_truth(tmp_path, "nba_test", rows)
    truth = load_truth(tmp_path, "nba_test")
    assert truth["testo sampleson"]["2019-20"] == 450
    # tamper with the frozen key -> loading must refuse
    csv_path.write_text(csv_path.read_text().replace("450", "451"))
    with pytest.raises(RuntimeError):
        load_truth(tmp_path, "nba_test")


def test_exact_grading_math():
    truth = {"2019-20": 450, "2020-21": 612, "2021-22": 388}
    asserted = {"2019-20": 450,          # exact
                "2020-21": 600,          # season hit, wrong value
                "2023-24": 100}          # spurious
    g = grade("p", truth, asserted)
    assert g.n_truth == 3 and g.n_season_hit == 2 and g.n_matched_exact == 1
    assert g.n_spurious == 1
    assert abs(g.record_recall - 2 / 3) < 1e-9
    assert abs(g.exact_recall - 1 / 3) < 1e-9
    assert g.true_total == 1450 and g.asserted_total == 1150
    assert abs(g.rel_agg_error - 300 / 1450) < 1e-9
