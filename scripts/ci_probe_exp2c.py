"""CI entry point for the Exp 2-C probe (dispatch: run-sports-script,
branch=citation). Arms the network gate — CI runs are a deliberate
dispatch — runs scripts/probe_enumerators.py with its defaults, and makes
sure a crash still leaves a readable report in data/ci/."""
import os
import sys
import traceback
from pathlib import Path

os.environ["RUN_ONLINE"] = "1"
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_enumerators  # noqa: E402

try:
    probe_enumerators.main()
except BaseException:                      # report the failure, then fail CI
    out = probe_enumerators.OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write("\n## PROBE CRASHED\n```\n" + traceback.format_exc() + "```\n")
    raise
