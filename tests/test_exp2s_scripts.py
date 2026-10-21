"""The actual experiment scripts, run as subprocesses against a synthetic
cohort in an isolated WEBAGG_DATA_DIR — the scripts' real CLI path, including
the import-time config, the freeze-hash check and the loud refusal gates."""
import csv
import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import seed_selftest_cohort

REPO = Path(__file__).resolve().parents[1]


def _run(script, *args, data_dir, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k != "RUN_LIVE"}
    env["WEBAGG_DATA_DIR"] = str(data_dir)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],
                          capture_output=True, text=True, cwd=REPO, env=env)


def test_replay_and_grade_scripts_end_to_end(tmp_path):
    cohort = seed_selftest_cohort(tmp_path)

    r = _run("replay_exp2s.py", "--cohort", cohort, data_dir=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Self-tests S1/S2 passed" in r.stdout
    res = tmp_path / "experiments" / "exp2s" / "results"
    stops = list(csv.DictReader(open(res / "stops.csv")))
    assert len(stops) == 2 * 4                   # 2 players x 4 eps arms
    kinds = {(row["player"], row["eps"]): row["stop_kind"] for row in stops}
    # Testo: checksum closure -> every arm stops 'checksum';
    # Benchy: no claims + toy-N psi -> every arm censored (honest)
    assert all(kinds[("testo sampleson", e)] == "checksum"
               for e in ("0.05", "0.1", "0.2", "0.3"))
    assert all(kinds[("benchy marks", e)] == "censored"
               for e in ("0.05", "0.1", "0.2", "0.3"))
    assert (res / "calibration.csv").exists() and (res / "shortfall.csv").exists()

    g = _run("grade_exp2s.py", "--cohort", cohort, data_dir=tmp_path)
    assert g.returncode == 0, g.stdout + g.stderr
    grades = {row["player"]: row for row in
              csv.DictReader(open(res / "grades.csv"))}
    assert grades["testo sampleson"]["exact_recall"] == "1.0"
    assert grades["testo sampleson"]["rel_agg_error"] == "0.0"
    # Benchy's third season is on no page: recall 2/3, and that's the number
    assert grades["benchy marks"]["record_recall"] == "0.6667"


def test_run_script_refuses_without_live_gate(tmp_path):
    seed_selftest_cohort(tmp_path)
    r = _run("run_exp2s.py", "--cohort", "nba_selftest", data_dir=tmp_path)
    assert r.returncode != 0
    assert "refusing live spend" in (r.stdout + r.stderr)


def test_run_script_refuses_without_keys_even_when_live(tmp_path):
    seed_selftest_cohort(tmp_path)
    env = {"RUN_LIVE": "1", "OPENAI_API_KEY": "", "SERPER_API_KEY": ""}
    r = _run("run_exp2s.py", "--cohort", "nba_selftest", data_dir=tmp_path,
             extra_env=env)
    assert r.returncode != 0
    assert "not set" in (r.stdout + r.stderr)
