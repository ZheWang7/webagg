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
    r = _run("run_exp2s.py", "--cohort", "nhl_selftest", data_dir=tmp_path)
    assert r.returncode != 0
    assert "refusing live spend" in (r.stdout + r.stderr)


def test_run_script_refuses_without_keys_even_when_live(tmp_path):
    seed_selftest_cohort(tmp_path)
    env = {"RUN_LIVE": "1", "OPENAI_API_KEY": "", "SERPER_API_KEY": ""}
    r = _run("run_exp2s.py", "--cohort", "nhl_selftest", data_dir=tmp_path,
             extra_env=env)
    assert r.returncode != 0
    assert "not set" in (r.stdout + r.stderr)


def test_collect_calibration_refuses_without_live_gate(tmp_path):
    r = _run("collect_calibration.py", "--player", "Testo Sampleson",
             data_dir=tmp_path)
    assert r.returncode != 0
    assert "refusing live spend" in (r.stdout + r.stderr)


def test_collect_calibration_refuses_to_overwrite(tmp_path):
    cal = tmp_path / "conformal" / "calibration.csv"
    cal.parent.mkdir(parents=True)
    cal.write_text("confidence,correct\n0.9,1\n")
    env = {"RUN_LIVE": "1", "OPENAI_API_KEY": "k", "SERPER_API_KEY": "k"}
    r = _run("collect_calibration.py", "--player", "Testo Sampleson",
             data_dir=tmp_path, extra_env=env)
    assert r.returncode != 0
    assert "must not be silently overwritten" in (r.stdout + r.stderr)


def test_collect_calibration_dump_and_gate_roundtrip(tmp_path):
    """dump_rows -> hand-label -> ConformalGate.load works end to end,
    including the extra context columns the loader must tolerate."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from collect_calibration import dump_rows
    from webaggs.conformal import ConformalGate
    from webaggs.storage import Ledger
    from webaggs.types import Mention, Source

    led = Ledger(tmp_path / "pilot.sqlite")
    for i in range(25):
        conf = 0.70 + i * 0.01
        led.add_source(Source(source_id=f"s{i}", url=f"https://w{i}.test/p",
                              domain=f"w{i}.test", fetch_time="t",
                              formulation_id="f", status="ok"))
        led.add_mention(Mention(mention_id=f"m{i}", source_id=f"s{i}",
                                entity_surface="Testo Sampleson",
                                season="2015-16", attribute="points",
                                value=40 + i, confidence=conf), accepted=True)
    led.commit()
    out = tmp_path / "conformal" / "calibration.csv"
    assert dump_rows(led, "pilot", out) == 25
    led.close()

    # hand-label: mark all correct (the loader only needs the two columns)
    import csv as _csv
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    assert rows and rows[0]["correct"] == ""          # shipped unlabeled
    for row in rows:
        row["correct"] = "1"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    gate = ConformalGate.load(out)
    assert gate.n_cal == 25 and 0.0 < gate.threshold <= 0.30 + 1e-9
