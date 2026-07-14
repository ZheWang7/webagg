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
MODEL_CHEAP = os.environ.get("WEBAGG_MODEL_CHEAP", "gpt-5.5-mini")
MODEL_STRONG = os.environ.get("WEBAGG_MODEL_STRONG", "gpt-5.5")

# --- stopping-rule / frontier constants (design doc §3) ---------------------
EPSILON = 0.10     # target unseen-mass threshold
DELTA = 0.10       # confidence parameter
ETA = 0.5          # frontier exploration parameter
MAX_STEPS = 200    # hard cap on agent steps

# --- politeness -------------------------------------------------------------
MAX_REQUESTS_PER_SEC_PER_DOMAIN = 1.0
