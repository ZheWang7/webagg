"""Single-provider LLM access (one API, one model — project design rule).

Lessons baked in from the finance codebase:
  * call_llm returns the model's parsed JSON payload DIRECTLY — no wrapper
    envelope. (The finance call_llm wrapped replies in {"payload": ...} and
    a consumer reading top-level keys silently discarded every reply.)
  * The raw reply text is ALWAYS persisted to the ledger when one is passed,
    so any downstream parsing failure is diagnosable from one row.
  * One retry on malformed JSON, then a LOUD exception — never a silent
    default (loud-failure-over-silent-degradation rule).
"""
from __future__ import annotations

import json
import os
import time

import requests

from . import config

_API_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def _post(system: str, user: str) -> str:
    key = os.environ.get(config.OPENAI_API_KEY_ENV)
    if not key:
        raise LLMError(f"{config.OPENAI_API_KEY_ENV} not set — refusing to run "
                       "(no silent degraded mode).")
    for attempt in (1, 2):
        resp = requests.post(
            _API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": config.LLM_MODEL,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "response_format": {"type": "json_object"}},
            timeout=90)
        if resp.status_code in (429, 500, 502, 503) and attempt == 1:
            time.sleep(15)                   # one paced retry on transient
            continue                         # errors, then loud failure —
        resp.raise_for_status()              # a blip must not forfeit a run
        return resp.json()["choices"][0]["message"]["content"]
    raise LLMError("unreachable")


def call_llm(system: str, user: str, *, ledger=None, run_id: str = "",
             purpose: str = "") -> dict:
    """Return the model's JSON reply as a dict. Raises LLMError after one retry."""
    last_raw = ""
    for attempt in (1, 2):                       # one retry, then loud failure
        raw = _post(system, user)
        last_raw = raw
        if ledger is not None:                   # raw reply persisted first
            ledger.log_llm(run_id, purpose, raw, config.LLM_MODEL)
            ledger.commit()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"LLM returned non-JSON twice (purpose={purpose}). "
                   f"Last raw reply (truncated): {last_raw[:300]!r}")


def load_prompt(name: str) -> str:
    """Prompts are anchored to the package dir, never the process CWD
    (the CWD-relative prompt trap crashed finance runs mid-spend)."""
    path = config.PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt missing: {path}")
    return path.read_text(encoding="utf-8")
