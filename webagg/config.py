"""
Configuration: loads .env and defines run-wide constants.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

# --- paths ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
HTML_CACHE_DIR = ROOT_DIR / "html_cache"
for _d in (RUNS_DIR, GROUND_TRUTH_DIR, HTML_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- secrets ----------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
USER_AGENT = os.environ.get("USER_AGENT", "webagg-research/0.1 (mailto:jameswangzhe1110@gmail.com)")

# --- LLM model keys (impl guide ch. 5) ---------------------------------------
# Cheap model for high-volume yes/no work (relevance, ER adjudication);
# stronger model for structured extraction. Both overridable via .env.
MODEL_CHEAP = os.environ.get("WEBAGG_MODEL_CHEAP", "gpt-5-nano")
MODEL_STRONG = os.environ.get("WEBAGG_MODEL_STRONG", "gpt-5")

# --- reader gate / audit constants (impl guide ch. 6) ------------------------
DELTA_E = 0.05      # conformal miscoverage level (paper Prop. 2)
DELTA_A = 0.05      # confidence level of the phi-audit Clopper-Pearson bound
CALIBRATION_SET = DATA_DIR / "calibration" / "extraction_cal.json"

# --- stopping-rule / frontier constants (impl guide ch. 7, paper §3.3) --------
EPS_G = 0.10       # per-stratum unseen-mass threshold eps_g (conjunct i)
DELTA_M = 0.10     # confidence budget for the psi radii (union-bounded via w_g)
ETA = 0.20         # "hot frontier" residual-yield threshold, in records (conjunct ii)
MAX_STEPS = 200    # hard cap on agent steps (= max capture occasions)
Y_CAP = 12         # per-occasion novelty cap (paper Assumption (b))
BETA = 1.0         # frontier-credit weight in U_hat
LAMBDA_PER_RECORD = 0.50   # $ value of one new record (paper App. B; sane at Serper prices)
SEARCH_COST_USD = 0.02     # $ per search issuance
BUDGET_USD = 5.0           # default per-run spend cap
# Optional refinements (paper appendices), behind flags per guide §7:
USE_CHAO_BRAKE = False       # App. C capture-recapture brake (can only FORBID stopping)
USE_ECONOMIC_ORDER = True    # App. B reservation-index ordering + economic stop

# --- claims engine / checksums (impl guide §11, design paper §5.2) -----------
CLAIM_TOL_REL = 0.02           # SUM certifies when Delta+ <= CLAIM_TOL_REL * V
CLAIM_BRAKE_MIN_BELIEF = 0.50  # corroborated-COUNT belief needed to arm the App. E
CLAIM_SCOPE_FORBID = ("debt", "to date")   # scope words that DEMOTE a claim (never certify)
CLAIM_SCOPE_REQUIRE = ("equity", "round")  # a STATED scope must contain one of these

# Backward-compat aliases (pre-SIGMOD names; do not use in new code)
EPSILON = EPS_G
DELTA = DELTA_M

# --- politeness -------------------------------------------------------------
MAX_REQUESTS_PER_SEC_PER_DOMAIN = 1.0
