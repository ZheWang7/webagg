"""Truth freeze / exact grading (domain-free); oracle driver schema tests
live in test_oracle_hardening.py."""
import pytest

from webaggs.truth import grade, load_truth, write_truth


def test_truth_freeze_detects_tampering(tmp_path):
    rows = [{"player": "Testo Sampleson", "player_id": 1,
             "season": "2019-20", "pts": 450, "gp": 60}]
    csv_path = write_truth(tmp_path, "nhl_test", rows)
    truth = load_truth(tmp_path, "nhl_test")
    assert truth["testo sampleson"]["2019-20"] == 450
    # tamper with the frozen key -> loading must refuse
    csv_path.write_text(csv_path.read_text().replace("450", "451"))
    with pytest.raises(RuntimeError):
        load_truth(tmp_path, "nhl_test")


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
