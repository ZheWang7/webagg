"""The exp2c experiment scripts as real subprocesses against a synthetic
cohort in an isolated WEBAGG_DATA_DIR."""
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
    return subprocess.run([sys.executable, str(REPO / "scripts" / script),
                           *args], capture_output=True, text=True, cwd=REPO,
                          env=env)


def test_replay_and_grade_scripts_end_to_end(tmp_path):
    cohort = seed_selftest_cohort(tmp_path)
    r = _run("replay_exp2c.py", "--cohort", cohort, "--live-eps", "0.30",
             "--arms", "0.5,0.4,0.3", data_dir=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Self-tests S1/S2 passed" in r.stdout
    res = tmp_path / "experiments" / "exp2c" / "results"
    stops = list(csv.DictReader(open(res / "stops.csv")))
    assert len(stops) == 2 * 3                     # 2 papers x 3 arms
    kinds = {(row["player"][:20], row["eps"]): row["stop_kind"]
             for row in stops}
    big = [v for (p, _e), v in kinds.items() if p.startswith("big sample")]
    small = [v for (p, _e), v in kinds.items() if p.startswith("testo")]
    assert set(big) == {"statistical"}             # THE path, at every arm
    assert set(small) == {"censored"}              # tiny n: never certifies

    g = _run("grade_exp2c.py", "--cohort", cohort, data_dir=tmp_path)
    assert g.returncode == 0, g.stdout + g.stderr
    grades = {row["player"][:20]: row for row in
              csv.DictReader(open(res / "grades.csv"))}
    assert grades["big sample transform"]["record_recall"] == "1.0"
    assert grades["big sample transform"]["certificate"] == "statistical"
    assert grades["testo attention nets"]["record_recall"] == "1.0"


def test_live_scripts_refuse_without_gate(tmp_path):
    seed_selftest_cohort(tmp_path)
    for script, args in (("run_exp2c.py", ["--cohort", "cite_selftest",
                                           "--eps", "0.30"]),
                         ("pilot_ycap.py", ["--paper", "Some Pilot Paper"])):
        r = _run(script, *args, data_dir=tmp_path)
        assert r.returncode != 0
        assert "refusing live spend" in (r.stdout + r.stderr).lower() or \
            "RUN_LIVE" in (r.stdout + r.stderr)
