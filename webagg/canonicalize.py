from __future__ import annotations

import re

# scale words -> multiplier ("40M" -> 40 * 1e6)
_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}

# a money/number spelling: optional currency, digits (commas ok), optional
# scale word, optional trailing currency word. fullmatch only -- a value that
# merely CONTAINS a number ("Series B") must not be treated as a number.
_NUM = re.compile(
    r"(?:usd|us\$|\$|\u20ac|\u00a3)?\s*"   # leading currency marker
    r"(\d[\d,]*(?:\.\d+)?)"                # 40 | 40,000,000 | 0.04
    r"\s*([a-z]+)?"                        # optional scale word: m / million / bn
    r"\s*(?:usd|dollars?)?",               # optional trailing currency word
    re.I,
)

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def canonicalize_value(raw: str) -> str:
    """Map every spelling of one value to a single canonical string.

    Numbers/amounts -> base units with no separators ("40000000"), enforcing
    what prompts/extract.txt already asks the LLM for. ISO dates pass through
    untouched (they are already canonical record keys, e.g. the (date, opponent)
    key in the LeBron ground truth). Everything else is case/whitespace
    normalized so "Indiana  Pacers" and "indiana pacers" group as ONE value.
    """
    s = (raw or "").strip()
    if not s:
        return s

    # 1. dates are already canonical -- do not lowercase or reformat them
    if _ISO_DATE.fullmatch(s):
        return s

    # 2. money / plain numbers -> base units
    m = _NUM.fullmatch(s)
    if m:
        num, scale = m.group(1), (m.group(2) or "").lower()
        if scale == "" or scale in _SCALE:
            v = float(num.replace(",", "")) * _SCALE.get(scale, 1.0)
            # integral values render without a trailing ".0" so that
            # "$40M" and "40000000" produce the identical string key
            return str(int(v)) if v.is_integer() else str(v)
        # unknown trailing word ("40 games") -> not a pure amount; fall through

    # 3. generic strings: collapse whitespace + lowercase for stable grouping
    return re.sub(r"\s+", " ", s).lower()
