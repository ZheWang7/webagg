"""CI entry point for the Exp 2-C cohort selection (dispatch:
run-sports-script, branch=citation). Arms the network gate — a CI run is
a deliberate dispatch — and runs scripts/select_cohort.py with the
chosen parameters (eps 0.6, tau_max 1150, cutoff 2023-12-31). The
outputs under cohort/ are committed back by the workflow."""
import os
import sys
from pathlib import Path

os.environ["RUN_ONLINE"] = "1"
# u range: at tau_max 1150 the widest singleton-share range with a usable
# band is about [0.22, 0.30]; coverage 0.80 keeps papers whose Europe PMC
# share is close to the probe paper's (0.83, where u was ~0.28)
sys.argv = [sys.argv[0], "--eps", "0.6", "--tau-max", "1150",
            "--cutoff", "2023-12-31", "--u-lo", "0.22", "--u-hi", "0.30",
            "--min-coverage", "0.80"]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_cohort  # noqa: E402

select_cohort.main()
