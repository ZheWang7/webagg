"""Cohort screening: candidate names -> CIK-resolved Form D filers .

The proposal's shared setup wants 50-150 companies whose FULL funding
history is in EDGAR Form D. This module does the MECHANICAL part of that
screen over a candidate-names file:

  1. NAME -> CIK. EDGAR's company_tickers.json (what the driver's
     name_contains sweeps) only lists TICKERED companies -- useless for the
     private filers the cohort mostly consists of. We use EDGAR's company
     browse endpoint instead (browse-edgar?action=getcompany&company=...
     &type=D&output=atom), which searches ALL filers and, with type=D,
     already restricts to companies that filed at least one Form D.
  2. STATS per resolved filer, from the submissions index (no XML fetches
     at screening time -- chains are build_truth.py's job): number of Form
     D / D/A filings, an original-D count as a rounds proxy, and the
     first/last filing dates.
  3. THE SHEET. One CSV row per candidate with an AUTO verdict:
        no_match   -- browse found no Form D filer resembling the name
        weak_match -- a filer was found but the name similarity is low;
                      a human must confirm the CIK before trusting it
        ok         -- confidently resolved, stats attached
     plus an empty human_verdict column.

  WHAT STAYS HUMAN (deliberately): the FULL-HISTORY judgment. Whether a
  company's earliest Form D predates its earliest press-known round is not
  mechanically decidable from EDGAR alone -- Uber resolves fine and has 8
  clean chains, yet its Series A/seed never touched Form D. The sheet
  carries the evidence (first_filing date vs what you know of the company);
  the reject decision is yours, recorded in human_verdict and kept in git.

Name matching: normalized token containment + difflib ratio against the
filer's registered name, with corporate suffixes stripped. Conservative
thresholds -- a wrong CIK silently poisons a truth table, so borderline
matches are flagged weak_match rather than auto-accepted.
"""
from __future__ import annotations

import csv
import difflib
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import httpx

from . import config

BROWSE_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
              "&company={q}&type=D&dateb=&owner=include&count=40"
              "&output=atom")
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ATOM = "{http://www.w3.org/2005/Atom}"

# corporate dressing that press names drop and registered names carry
_SUFFIX_RX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|lp|l\.p"
    r"|ltd|limited|holdings|technologies|technology|labs|systems|group"
    r"|the)\b\.?", re.I)


def norm_name(s: str) -> str:
    """Lowercase, strip suffixes/punctuation -> comparable core name."""
    s = _SUFFIX_RX.sub(" ", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def name_score(candidate: str, registered: str) -> float:
    """Similarity in [0,1] between a press name and a registered filer name.

    Token containment first (every candidate token present in the
    registered name scores high: "Databricks" vs "Databricks, Inc.");
    difflib ratio as the general fallback. Conservative by design.
    """
    c, r = norm_name(candidate), norm_name(registered)
    if not c or not r:
        return 0.0
    if c == r:
        return 1.0
    c_tok, r_tok = set(c.split()), set(r.split())
    if c_tok and c_tok <= r_tok:
        return 0.95                     # all candidate tokens present
    return difflib.SequenceMatcher(None, c, r).ratio()


def parse_browse_atom(xml_text: str) -> list[tuple[str, str]]:
    """browse-edgar Atom -> [(cik10, registered_name), ...].

    Two shapes exist: MULTIPLE matches (entries with <title> and a CIK in
    the entry links) and a SINGLE match (company-info block). Both handled.
    """
    root = ET.fromstring(xml_text)
    out: list[tuple[str, str]] = []
    # multi-match shape
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        href = ""
        link = entry.find(f"{_ATOM}link")
        if link is not None:
            href = link.get("href", "")
        m = re.search(r"CIK=(\d+)", href)
        if m and title:
            out.append((m.group(1).zfill(10), title))
    if out:
        return out
    # single-match shape: company-info carries cik + conformed-name
    for el in root.iter():
        if el.tag.endswith("company-info"):
            cik = name = None
            for ch in el.iter():
                if ch.tag.endswith("}cik") or ch.tag == "cik":
                    cik = (ch.text or "").strip()
                if (ch.tag.endswith("conformed-name")
                        or ch.tag == "conformed-name"):
                    name = (ch.text or "").strip()
            if cik and name:
                out.append((cik.zfill(10), name))
    return out


def formd_stats(subs: dict) -> dict:
    """Submissions JSON -> Form D stats (recent block; older pages of very
    prolific filers are irrelevant to a screen that only needs presence,
    counts, and the date span)."""
    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    d_dates = [d for f, d in zip(forms, dates) if f in ("D", "D/A")]
    return {
        "n_formd": len(d_dates),
        "n_original_d": sum(1 for f in forms if f == "D"),
        "first_formd": min(d_dates) if d_dates else "",
        "last_formd": max(d_dates) if d_dates else "",
    }


def read_candidates(path: Path) -> list[str]:
    """Candidate names file: one per line, '#' comments skipped."""
    return [ln.strip() for ln in Path(path).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


SHEET_COLUMNS = ["candidate", "verdict", "cik", "registered_name",
                 "match_score", "n_formd", "n_original_d",
                 "first_formd", "last_formd", "human_verdict", "note"]


def screen_one(name: str, client: httpx.Client, *,
               min_ok: float = 0.85, min_weak: float = 0.60,
               pause_s: float = 0.4) -> dict:
    """Screen ONE candidate name -> one sheet row (network: 1-2 requests)."""
    row = {c: "" for c in SHEET_COLUMNS}
    row["candidate"] = name
    r = client.get(BROWSE_URL.format(q=name.replace(" ", "+")))
    time.sleep(pause_s)                  # SEC politeness (10 req/s hard cap)
    if r.status_code != 200:
        row["verdict"] = "no_match"
        row["note"] = f"browse http {r.status_code}"
        return row
    matches = parse_browse_atom(r.text)
    if not matches:
        row["verdict"] = "no_match"      # no Form D filer resembles the name
        return row
    cik, reg = max(matches, key=lambda m: name_score(name, m[1]))
    score = name_score(name, reg)
    if score < min_weak:
        row["verdict"] = "no_match"
        row["note"] = f"best was {reg!r} ({score:.2f})"
        return row
    row.update(cik=cik, registered_name=reg, match_score=f"{score:.2f}",
               verdict="ok" if score >= min_ok else "weak_match")
    s = client.get(SUBMISSIONS_URL.format(cik=cik))
    time.sleep(pause_s)
    if s.status_code == 200:
        row.update({k: str(v) for k, v in formd_stats(s.json()).items()})
    else:
        row["note"] = f"submissions http {s.status_code}"
    return row


def write_sheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
