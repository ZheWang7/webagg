"""Canonicalization: typed validators for the reader (webaggr.pdf §7.1).

Look-alike numbers are the reader's main hazard; here the types are
player names, NBA seasons and integer point totals. Everything that
enters a record key or a grading comparison passes through here, so
extraction prompts, ER, truth build and grading all agree on one form.
"""
from __future__ import annotations

import re
import unicodedata

_SUFFIXES = (" jr", " sr", " ii", " iii", " iv")


def normalize_name(name: str) -> str:
    """Canonical player key: lowercase, ASCII-folded, no punctuation/suffixes.
    'Luka Dončić' -> 'luka doncic'; 'Jaren Jackson Jr.' -> 'jaren jackson'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    return s


_SEASON_RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–—/]\s*((?:19|20)?\d{2})\b")
_SEASON_SINGLE = re.compile(r"\b((?:19|20)\d{2})\b")


def canonical_season(text: str) -> str | None:
    """Canonical NBA season string 'YYYY-YY' (season starting year + end-year
    two digits): '2018-19', '2018–2019', '2018/19' all -> '2018-19'.
    A bare year 'YYYY' is read as the season STARTING that fall only when the
    end-year is absent — ambiguous forms are the reader's job to avoid, and
    the conformal gate prices what slips through (webaggr.pdf §7.1)."""
    m = _SEASON_RANGE.search(text)
    if m:
        start = int(m.group(1))
        end_raw = m.group(2)
        end = int(end_raw) if len(end_raw) == 4 else (start // 100) * 100 + int(end_raw)
        if end < start:                       # '1999-00' century rollover
            end += 100
        if end - start != 1:
            return None                       # '2018-2020' is a span, not a season
        return f"{start}-{end % 100:02d}"
    m = _SEASON_SINGLE.search(text)
    if m:
        start = int(m.group(1))
        return f"{start}-{(start + 1) % 100:02d}"
    return None


# must START with a digit: bare commas ("hello, world") are not numbers —
# the fuzzer caught [\d,]+ matching "," and crashing int("") mid-extraction
_POINTS = re.compile(r"-?\d[\d,]*(\.\d+)?")


def canonical_points(value) -> int | None:
    """Integer points. Rejects negatives, absurd magnitudes (career max ever
    is < 40,000 regular-season points — a typed bound) AND any fractional
    value: a decimal is a per-game figure, the look-alike number this
    validator exists to stop (webaggr.pdf §7.1) — truncating 28.4 to 28
    would silently assert a bogus season total."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None                       # 28.4 is PPG, not a total
        n = int(value)
    else:
        m = _POINTS.search(str(value))
        if not m:
            return None
        if m.group(1):                        # matched a decimal part: PPG
            return None
        n = int(m.group(0).replace(",", ""))
    if n < 0 or n > 60000:
        return None
    return n
