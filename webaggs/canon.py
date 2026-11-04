"""Canonicalization: typed validators for the reader (webaggr.pdf §7.1).

Citation domain (Exp 2-C): the record key is a citing paper's identity —
DOI when present, else a normalized title — produced identically on the
truth side (oracle_openalex) and the extraction side, so grading joins
exactly. The NHL-era validators below it are kept: they are generic string
functions and the sports branch's tests still cover them.

Look-alike numbers are the reader's main hazard; here the types are
player names, NHL seasons and integer point totals. Everything that
enters a record key or a grading comparison passes through here, so
extraction prompts, ER, truth build and grading all agree on one form.
"""
from __future__ import annotations

import re
import unicodedata

from . import config

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
    """Canonical NHL season string 'YYYY-YY' (season starting year + end-year
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
    """Integer points. Rejects negatives, absurd magnitudes (the NHL career
    record is 2,857 points, so the bound is config.POINTS_MAX) AND any
    fractional value: a decimal is a per-game figure, the look-alike number
    this validator exists to stop (webaggr.pdf §7.1) — truncating 1.4 to 1
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
    if n < 0 or n > config.POINTS_MAX:
        return None
    return n


# --- citation-domain validators (Exp 2-C) --------------------------------
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


def canonical_doi(text) -> str | None:
    """A DOI anywhere in the text -> lowercase bare DOI ('10.x/y'), with
    URL prefixes and trailing punctuation stripped; None if absent."""
    if not text:
        return None
    m = _DOI.search(str(text))
    if not m:
        return None
    doi = m.group(1).lower().rstrip(".,;:)]}>\"'")
    return doi if "/" in doi and len(doi) >= 8 else None


def normalize_title(title: str) -> str:
    """Normalized paper title for keying: accent-folded (NFKD, combining
    marks stripped — so é joins e), casefolded, Unicode punctuation removed
    but NON-LATIN SCRIPTS PRESERVED (the citation graph is full of CJK and
    Cyrillic titles; ASCII-folding erased them — found live by the driver's
    unkeyable-citer refusal on W37211910). Same function on both sides, so
    truth and assertions join whenever the characters do."""
    s = unicodedata.normalize("NFKD", str(title or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold()
    s = re.sub(r"[^\w\s]", " ", s)          # \w is Unicode-aware in py3
    s = re.sub(r"_", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160].strip()


def cite_key_from_parts(doi, title) -> str | None:
    """The grading/record key: 'doi:<doi>' when a DOI exists, else
    't:<normalized title>' when the title has at least 3 words (shorter is
    too collision-prone to key on); None when neither works."""
    d = canonical_doi(doi)
    if d:
        return f"doi:{d}"
    t = normalize_title(title)
    # keyable: >= 3 tokens (spaced scripts) OR >= 6 characters ignoring
    # spaces (CJK and other unspaced scripts carry whole words per glyph)
    if t and (len(t.split()) >= 3 or len(t.replace(" ", "")) >= 6):
        return f"t:{t}"
    return None
