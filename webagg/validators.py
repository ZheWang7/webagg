"""Typed validators (impl guide §6.2): the extraction confusion suite.

Stage 3 of the four-stage gate. Deterministic, zero-cost checks that run
BEFORE any probability is spent (paper §7.1, Table 3). Each check either
canonicalizes, flags (appended to Mention.validator_flags), or rejects by
raising Reject. The suite targets the confusions that dominate reader
error in deployment:

    round vs. cumulative   ("$63M to date")
    raise vs. post-money   ("valuing the company at $300M")
    currency               ("EUR 37M ($40M)")
    extension vs. new round("Series B extension")
    magnitude              ("$40MM", "0.04B")
    date roles             (announced / closed / filed)

Relation to canonicalize.py: canonicalize_value() produces the STRING
grouping key that corroboration buckets on; canonicalize_money() here
produces the TYPED numeric (Mention.value_num, base-unit USD) plus flags.
Both share one scale table (single-definition discipline, guide §4.3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .type_defs import Mention
from .canonicalize import _SCALE          # ONE scale table for the project


class Reject(Exception):
    """Fatal validation failure: the mention must not reach the gate.

    The reason string is recorded in validator_flags on the (kept but
    rejected) mention so the phi-audit and error analysis can see WHY.
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


# ---------------------------------------------------------------------------
# Extraction context: cross-mention facts the validators may consult.
# Filled by the caller with whatever is already corroborated for this
# entity; every field is optional -- validators degrade gracefully.
# ---------------------------------------------------------------------------
@dataclass
class ExtractionContext:
    stage: str | None = None          # "seed" | "series a" | ... (lowercased)
    post_money: float | None = None   # corroborated post-money valuation, USD
    cumulative: float | None = None   # corroborated raised-to-date, USD


# ---------------------------------------------------------------------------
# Money canonicalization (the workhorse, guide §6.2 listing)
# ---------------------------------------------------------------------------
_AMOUNT_RX = re.compile(
    r"(?:usd|us\$|\$|\u20ac|\u00a3|eur|gbp)?\s*"   # leading currency marker
    r"(\d[\d,]*(?:\.\d+)?)"                        # 40 | 40,000,000 | 0.04
    r"\s*([a-z]+)?",                               # scale word: m/mm/bn/million
    re.I,
)

# PROVISIONAL static FX table (flagged on every use). The guide wants the
# cross-check "at t_asof", which needs a dated-rate source; until the
# registry/data chapter provides one, we convert with a coarse static rate
# and flag, so downstream can see the value crossed a currency boundary.
_FX_STATIC = {"EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CHF": 1.12,
              "CAD": 0.73, "AUD": 0.66, "CNY": 0.14, "USD": 1.0}


def parse_amount(raw: str) -> tuple[float, str]:
    """'0.04B' -> (0.04, 'b');  '$40MM' -> (40.0, 'mm');  '40,000,000' -> (4e7, '').

    Raises Reject on anything that is not a single money/number spelling.
    """
    m = _AMOUNT_RX.fullmatch((raw or "").strip())
    if not m:
        raise Reject("amount_unparseable")
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit and unit not in _SCALE:
        raise Reject("amount_unit_unknown")      # "40 games" is not money
    return num, unit


def fx_rate(currency: str, target: str, t_asof: datetime | None) -> float:
    """PROVISIONAL: static rate, ignores t_asof (see _FX_STATIC note)."""
    try:
        return _FX_STATIC[currency.upper()] / _FX_STATIC[target.upper()]
    except KeyError:
        raise Reject(f"fx_unknown_currency:{currency}")


def canonicalize_money(raw: str, currency: str | None,
                       t_asof: datetime | None) -> tuple[float, list[str]]:
    """Money string -> (base-unit USD float, flags). Guide §6.2 workhorse.

    A money attribute without a currency tag is REJECTED, not defaulted:
    silently assuming USD is exactly the EUR/USD confusion the suite exists
    to catch.
    """
    flags: list[str] = []
    num, unit = parse_amount(raw)
    if currency is None:
        raise Reject("currency_missing")         # money attrs MUST carry currency
    usd = num * _SCALE.get(unit, 1.0)
    if currency.upper() != "USD":
        usd *= fx_rate(currency, "USD", t_asof)
        flags.append(f"fx:{currency.upper()}")
        flags.append("fx_static_rate")           # PROVISIONAL rate, see above
    return usd, flags


# ---------------------------------------------------------------------------
# Cue lexicons + plausibility bands (guide §6.2 table rows)
# ---------------------------------------------------------------------------
CUE_CUMULATIVE = re.compile(
    r"\bto\s+date\b|\btotal\s+(?:raised|funding)\b|\bhas\s+raised\b"
    r"|\bbringing\s+(?:its\s+)?total\b|\bcumulative\b|\braised\s+so\s+far\b", re.I)

EXTENSION_RX = re.compile(
    r"\bextension\b|\bextended\s+(?:its\s+)?series\b|\btop[- ]?up\b"
    r"|\bsecond\s+close\b|\bfollow[- ]?on\s+tranche\b", re.I)

# Generous per-stage plausibility bands, USD (flag-only, never reject:
# outliers exist; the band catches magnitude/unit slips like 0.04B vs 40B).
_STAGE_BANDS: dict[str, tuple[float, float]] = {
    "pre-seed": (5e4, 3e6),
    "seed": (1e5, 1.5e7),
    "series a": (2e6, 8e7),
    "series b": (1e7, 3e8),
    "series c": (2e7, 8e8),
}


def stage_plausible(stage: str | None, value_num: float | None) -> bool:
    """True when we cannot say the amount is implausible for the stage."""
    if stage is None or value_num is None:
        return True                              # nothing to check against
    band = _STAGE_BANDS.get(stage.lower())
    return band is None or band[0] <= value_num <= band[1]


# ---------------------------------------------------------------------------
# Per-kind validators (return flags; raise Reject on fatal)
# ---------------------------------------------------------------------------
def validate_round(m: Mention, ctx: ExtractionContext) -> list[str]:
    """Confusion-suite rows for funding-round amount mentions."""
    flags: list[str] = []
    # round vs. cumulative: "$63M to date" is a Claim about the stratum,
    # not a round amount -- fatal for a round-amount mention.
    if CUE_CUMULATIVE.search(m.passage):
        raise Reject("cumulative_not_round")
    if ctx.cumulative and m.value_num and m.value_num > ctx.cumulative * 1.001:
        # a single round cannot exceed the corroborated raised-to-date
        raise Reject("round_exceeds_cumulative")
    # extension vs. new round: flag; ER/dedup decides using date proximity
    if EXTENSION_RX.search(m.passage):
        flags.append("series_extension")
    # raise vs. post-money: amounts near the corroborated valuation are
    # suspiciously likely to BE the valuation
    if ctx.post_money and m.value_num and m.value_num > 0.8 * ctx.post_money:
        flags.append("amount_near_postmoney")
    if not stage_plausible(ctx.stage, m.value_num):
        flags.append("magnitude_outlier")
    return flags


def validate_date(m: Mention) -> list[str]:
    """Date-role row: a date without a role is ambiguous by construction
    (announced vs. closed vs. filed differ by weeks). Flag, don't reject --
    the filing-window consistency check needs authority-chain context and
    arrives with the registry chapter."""
    return [] if m.date_role else ["date_role_missing"]


_MONEY_ATTRS = {"amount", "post_money", "valuation", "raised"}
_DATE_ATTRS = {"date", "announced", "closed", "filed"}


def validate_mention(m: Mention, ctx: ExtractionContext | None = None) -> Mention:
    """Run the applicable suite rows on one mention (stage 3 of the gate).

    Returns the mention with value_num/currency canonicalized and
    validator_flags extended. Raises Reject on fatal rows; the CALLER
    records the rejection (the mention is kept with accepted=False, never
    deleted -- audit material, guide §6.1).
    """
    ctx = ctx or ExtractionContext()
    flags: list[str] = []
    if m.attribute in _MONEY_ATTRS:
        m.value_num, money_flags = canonicalize_money(m.value, m.currency, m.t_asof)
        flags += money_flags
        if m.record_kind == "funding_round" and m.attribute == "amount":
            flags += validate_round(m, ctx)
    elif m.attribute in _DATE_ATTRS:
        flags += validate_date(m)
    m.validator_flags = list(m.validator_flags) + flags
    return m
