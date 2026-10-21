"""Central configuration for webagg-sports (Exp 2-S, NBA domain).

Parameter names and defaults follow webaggr.pdf and the implementation
guide's parameter table (Table 4). Values changed for Exp 2-S are noted.
"""
import os
from pathlib import Path

# Anchor everything to the package, never the process CWD
# (prompts opened relative to CWD crash mid-run; see methodology notes).
ROOT_DIR = Path(__file__).resolve().parent          # .../webaggs
PROJECT_DIR = ROOT_DIR.parent                       # repo root
PROMPTS_DIR = ROOT_DIR / "prompts"
# Overridable so tests and pilot runs never touch the real experiment data.
DATA_DIR = Path(os.environ.get("WEBAGG_DATA_DIR", str(PROJECT_DIR / "data")))

# --- Stopping rule (webaggr.pdf §3.2-3.3) -------------------------------
EPS_G = 0.05        # per-group unseen-mass target: Exp 2-S live arm runs at 0.05
DELTA_M = 0.05      # stopping deviation budget (anytime-valid)
ETA = 0.20          # frontier-exhaustion threshold on residual yield
BETA = 1.0          # frontier-credit weight in U_hat (eq. 1)
Y_CAP = 12          # per-occasion novelty cap = increment bound (Assumption 1b)
MAX_STEPS = 200     # tau_max: budget cap on capture occasions
SEED_DEPTH = 12     # seed formulations per run (production seeder depth)

# --- Economics (webaggr.pdf App. B) -------------------------------------
LAMBDA_PER_RECORD = 0.50   # $ value of one new record; 0 once all strata certified
SEARCH_COST = 0.02         # $ per search issuance (Serper)
BUDGET_USD = 5.0           # per-entity search budget

# --- Reader / fidelity (webaggr.pdf §7) ---------------------------------
DELTA_E = 0.05      # conformal extraction miscoverage
TOL_REL = 0.02      # checksum relative tolerance (SUM claims)
# eps_F is a per-domain calibrated constant (impl guide §13.4). The NBA
# domain has NO calibrated eps_F yet; reports must say so loudly.
EPS_F_CALIBRATED = None

# --- Source-class reliability priors (webaggr.pdf §4) -------------------
# NBA domain analogue of the regulatory/vendor/news/blog ladder.
CLASS_PRIORS = {
    "stats_vendor": 0.80,   # basketball-reference, espn, statmuse, landofbasketball
    "news": 0.60,           # press / sports media
    "other": 0.40,          # blogs, forums, everything else
}

# --- Oracle deny list (PREREGISTRATION_EXP2S.md) -------------------------
# Official NBA data defines truth and is DENIED at run time: the
# withheld-registry design (webaggr.pdf §9). Everything else is "the web".
DENY_DOMAINS = ("nba.com", "stats.nba.com")

# --- LLM (single-provider design: one API, one model) --------------------
LLM_MODEL = os.environ.get("WEBAGG_LLM_MODEL", "gpt-5-mini")
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
SERPER_API_KEY_ENV = "SERPER_API_KEY"

# Live-spend gate: no experiment path runs live unless RUN_LIVE=1.
def live_enabled() -> bool:
    return os.environ.get("RUN_LIVE", "0") == "1"
