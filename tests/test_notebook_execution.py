"""Execute the Exp 2-S notebook headlessly, start to finish, against the
isolated self-test cohort — the notebook itself is part of the pytest gate.
The live cell must skip (no RUN_LIVE), every offline cell must run clean,
and the in-notebook self-tests S1/S2 must pass."""
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tests.conftest import seed_selftest_cohort

REPO = Path(__file__).resolve().parents[1]
NB_PATH = REPO / "notebooks" / "exp2s_stopping_calibration.ipynb"


@pytest.mark.slow
def test_notebook_executes_offline_end_to_end(tmp_path, monkeypatch):
    cohort = seed_selftest_cohort(tmp_path)
    # the kernel inherits this env: isolated data dir, pilot cohort, no live
    monkeypatch.setenv("WEBAGG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WEBAGG_COHORT", cohort)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("RUN_LIVE", raising=False)

    nb = nbformat.read(NB_PATH, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()                      # raises CellExecutionError on failure

    text = "\n".join(
        "".join(out.get("text", "") for out in cell.get("outputs", [])
                if out.get("output_type") == "stream")
        for cell in nb.cells if cell.cell_type == "code")
    assert "live phase skipped" in text          # the gate held
    assert "Self-tests S1/S2 passed" in text     # replay reproduces the stops
    assert "HEADLINE (statistical path" in text  # C2 table printed
    res = tmp_path / "experiments" / "exp2s" / "results"
    for f in ("stops.csv", "calibration.csv", "shortfall.csv", "grades.csv"):
        assert (res / f).exists(), f"notebook did not write {f}"
