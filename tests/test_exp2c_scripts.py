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
    r = _run("replay_exp2c.py", "--cohort", cohort, "--live-eps", "0.5",
             "--arms", "0.9,0.7,0.5", data_dir=tmp_path)
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
                                           "--eps", "0.5",
                                           "--max-steps", "600"]),
                         ("pilot_walk.py", ["--title", "Some Pilot Paper",
                                            "--doi", "10.1234/pilot.1",
                                            "--cutoff", "2023-12-31"])):
        r = _run(script, *args, data_dir=tmp_path)
        assert r.returncode != 0
        assert "refusing live spend" in (r.stdout + r.stderr).lower() or \
            "RUN_LIVE" in (r.stdout + r.stderr)


def test_truth_build_freezes_truth_targets_and_report(tmp_path, monkeypatch):
    """build_citation_truth.py end to end with a FAKE oracle driver:
    truth.csv is hash-locked, targets.json carries what the live run needs
    (title/DOI/PMID + cutoff), and run_exp2c's loader reads it back."""
    import importlib
    import json

    monkeypatch.setenv("WEBAGG_DATA_DIR", str(tmp_path))
    from webaggs import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    sys.path.insert(0, str(REPO / "scripts"))
    build = importlib.import_module("build_citation_truth")
    run = importlib.import_module("run_exp2c")
    monkeypatch.setattr(run, "config", config)

    class FakeDriver:
        def citers(self, doi, cutoff):
            assert cutoff == "2023-12-31"
            rows = [{"key": f"doi:10.5555/c.{i}", "doi": f"10.5555/c.{i}",
                     "year": 2020} for i in range(3)]
            return rows, {"raw_rows": 5, "no_doi": 1, "no_date": 0,
                          "after_cutoff": 1, "duplicate": 0, "multi_doi": 0,
                          "truth_rows": 3}

    papers = tmp_path / "papers.txt"
    papers.write_text("# comment\nA Landmark Paper | https://doi.org/10.1234/"
                      "LAND.1 | 555\nSecond Paper Here | 10.1234/second.2\n")
    monkeypatch.setattr(sys, "argv", ["x", "--cohort", "c1", "--papers",
                                      str(papers), "--cutoff", "2023-12-31"])
    build.main(driver=FakeDriver())

    from webaggs.truth import load_truth
    truth = load_truth(tmp_path, "c1")              # verifies the freeze hash
    assert set(truth) == {"a landmark paper", "second paper here"}
    assert set(truth["a landmark paper"]) == {f"doi:10.5555/c.{i}"
                                              for i in range(3)}
    tj = json.loads((tmp_path / "truth" / "c1" / "targets.json").read_text())
    assert tj["cutoff"] == "2023-12-31" and tj["oracle"] == "opencitations"
    targets, cutoff = run.load_targets("c1")
    assert cutoff == "2023-12-31"
    assert targets["a landmark paper"].doi == "10.1234/land.1"
    assert targets["a landmark paper"].pmid == "555"
    assert targets["second paper here"].pmid is None
    rep = list(csv.DictReader(open(tmp_path / "truth" / "c1"
                                   / "cohort_report.csv")))
    assert rep[0]["truth_rows"] == "3" and rep[0]["no_doi"] == "1"


def test_truth_build_rejects_bad_paper_lines():
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib

    import pytest
    build = importlib.import_module("build_citation_truth")
    for bad in ("Title Only", "Title | not-a-doi", "T | 10.1234/x.1 | abc"):
        with pytest.raises(SystemExit):
            build.parse_paper_line(bad)


def test_cohort_band_is_robust_to_the_singleton_share():
    """select_cohort.truth_band: the window must let the certificate fire
    for EVERY u in [u_lo, u_hi] — psi cleared at the worst u, both walks
    fitting inside tau_max at the most page-hungry u."""
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib

    import pytest

    from webaggs import design
    sc = importlib.import_module("select_cohort")
    t_lo, t_hi, n_lo, n_hi = sc.truth_band(0.7, 1000, 50, 0.20, 0.36,
                                           1.08, 0.02)
    assert t_lo < t_hi and n_lo < n_hi
    for u in (0.20, 0.28, 0.36):
        lo, hi = design.feasible_band(0.7, 50, 1000, u=u)
        assert lo <= n_lo and n_hi <= hi          # inside every u's band
    # truth window = N window scaled by N/truth, shrunk by the margin
    assert t_lo == pytest.approx(n_lo / 1.08 * 1.02)
    assert t_hi == pytest.approx(n_hi / 1.08 * 0.98)
    with pytest.raises(SystemExit):                # no robust band: say so
        sc.truth_band(0.7, 300, 50, 0.20, 0.36, 1.08, 0.02)
