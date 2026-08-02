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

# --- fragmentation: scan-vs-join routing (impl guide §12, paper App. D) ------
GAMMA_SCAN = 0.90          # single-class sufficiency: prune when >= this fraction
                           # of records are scan-sufficient under ONE class (Cor. 2)
MIN_RECORDS_FOR_PRUNE = 8  # REPO DEVIATION (documented in fragmentation.py): the
                           # guide states no floor, but gamma over 1-2 records
                           # fires trivially (1/1 = 100%). Waiting is conservative:
                           # a late prune costs only fetches, never the answer.

# Backward-compat aliases (pre-SIGMOD names; do not use in new code)
EPSILON = EPS_G
DELTA = DELTA_M

# --- politeness -------------------------------------------------------------
MAX_REQUESTS_PER_SEC_PER_DOMAIN = 1.0

# --- fidelity certificate ---------
# The completeness/fidelity deviation split of Theorem 6: the two-term
# interval holds with probability >= 1 - (DELTA_C + DELTA_F).
DELTA_C = 0.05        # completeness deviation share
DELTA_F = 0.05        # fidelity deviation share (Learn-Then-Test level)
EPS_F_TARGET = 0.10   # the fidelity level we ATTEMPT to certify; the stored
# certificate records what was actually achieved
FIDELITY_CERT_DIR = DATA_DIR / "fidelity_certs"   # one JSON per domain
FIDELITY_CERT_DIR.mkdir(parents=True, exist_ok=True)

# --- end-to-end report layer (impl guide §14, paper Thm 6 / Cor. 1) ----------
# The interval printed to the user is TWO terms: eps_C^g (per-group
# completeness slack, regime-dependent) + eps_F (one domain-wide fidelity
# level from §13). If no fidelity certificate exists for the domain, §14's
# decision (risk_control.load_fidelity_cert docstring poses it) is: fall
# back to this conservative constant AND SAY SO in the regime label,
# rather than refuse to print an interval. Calibrating (§13) replaces it.
EPS_F_FALLBACK = 0.15

# Optional post-certification reliability refinement (paper §4.4 / App. F):
# values in checksum/registry-certified strata act as labels; each source's
# q becomes a closed-form Beta-posterior over its agreement rate. OFF by
# default -- the fixed-prior QTable is the guaranteed path; this is a
# drop-in refinement required by no theorem.
USE_CERTIFIED_REFINE = False
REFINE_PRIOR_STRENGTH = 4.0    # pseudo-count mass anchoring the class prior

# --- verification allocator (impl guide §14.3, "spending a human wisely") ----
VERIFY_BUDGET = 5          # top-B human checks to print (B <= 10: greedy fine)
VERIFY_BELIEF_FLOOR = 0.70 # adopted values below this belief become candidates
DELTA_T_VERIFY = 0.10      # weight on supersession checks (share of value at risk
                           # if the wrong version of a corrected figure was adopted)

# The pre-committed lambda grid for Learn-Then-Test, CHEAPEST-FIRST.
# Each lambda bundles the knobs that trade fidelity against abstention/cost:
# ER thresholds tau+/-, conformal level delta_E, reliability cap qbar.
# Order matters and is frozen BEFORE calibration (fixed-sequence testing);
# re-ordering after seeing losses voids the certificate (guide pitfall 8).
LTT_GRID = [
    {"tau_plus": 0.85, "tau_minus": 0.15, "delta_E": 0.05, "qbar": 0.30},  # paper defaults
    {"tau_plus": 0.90, "tau_minus": 0.10, "delta_E": 0.05, "qbar": 0.30},  # wider adjudication band
    {"tau_plus": 0.90, "tau_minus": 0.10, "delta_E": 0.02, "qbar": 0.30},  # + stricter reader gate
    {"tau_plus": 0.95, "tau_minus": 0.05, "delta_E": 0.02, "qbar": 0.20},  # most conservative
]
