"""The withheld oracle: OpenCitations Index (webaggr.pdf §9 role).

Endpoint: GET https://api.opencitations.net/index/v2/citations/doi:{DOI}
— an open, CC0 citation index (no key needed; 180 requests/minute per IP;
full dumps downloadable, so the truth is reproducible). Truth is DEFINED
by this API and DENIED to the agent at run time (config.DENY_DOMAINS
covers opencitations.net). This driver is used ONLY by truth-build /
cohort tooling, never by the pipeline.

Why OpenCitations is the oracle and not an enumerator: it returns the
whole citing list in ONE response. As a capture occasion that would be a
single page holding the entire population (Y_CAP = n, so the certificate
could never fire); as an oracle that is exactly what we want.

Scope rules it encodes (PREREGISTRATION_EXP2C.md rev 3):
  * the record universe is DOI-REGISTERED citing works: a row whose citing
    entity has no DOI is outside the universe (counted in the report);
  * records = citing works created ON OR BEFORE the frozen cutoff. Rows
    carry a `creation` date of year, year-month or full-date precision, so
    the cutoff must be a 31 December: then "creation year <= cutoff year"
    is exact at every precision. Rows with no creation date are excluded
    and counted;
  * record key: 'doi:<lowercased doi>' — the same key
    canon.cite_key_from_parts produces on the extraction side;
  * SELF-VALIDATION: the list length must equal the API's own
    /citation-count for the same DOI, else the driver refuses to emit
    truth — a truncated response can never look like data.
"""
from __future__ import annotations

import os
import re
import time

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout

from .canon import canonical_doi, cite_key_from_parts

_BASE = "https://api.opencitations.net/index/v2"
_UA = {"User-Agent": "webagg research client (Exp 2-C truth build)",
       "Accept": "application/json"}
_CUTOFF = re.compile(r"^(\d{4})-12-31$")


class OracleError(RuntimeError):
    pass


def citing_dois(citing_field: str) -> list[str]:
    """The `citing` field is a space-separated bag of ids for ONE work
    ('omid:br/06... doi:10.1/x pmid:123'). -> its canonical DOIs."""
    out = []
    for tok in str(citing_field or "").split():
        if tok.lower().startswith("doi:"):
            d = canonical_doi(tok[4:])
            if d and d not in out:
                out.append(d)
    return out


class OpenCitationsDriver:
    """Paced, retried once on transient errors, loud on everything else."""

    def __init__(self, pace_seconds: float = 0.5, client=None,
                 retry_cooldown: float = 10.0):
        self.pace = pace_seconds
        self.client = client or requests          # tests inject fakes
        self.retry_cooldown = retry_cooldown
        self._last = 0.0

    def _get(self, path: str):
        headers = dict(_UA)
        token = os.environ.get("OPENCITATIONS_ACCESS_TOKEN")
        if token:                                 # optional, polite
            headers["authorization"] = token
        wait = self.pace - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        detail = "no response"
        for attempt in (1, 2):
            self._last = time.time()
            try:
                resp = self.client.get(f"{_BASE}/{path}", headers=headers,
                                       timeout=(10, 300))   # big lists are slow
            except (ReqTimeout, ReqConnectionError) as e:
                detail = type(e).__name__
                if attempt == 1:
                    time.sleep(self.retry_cooldown)
                continue
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError:
                    detail = "200 but not JSON"
                    break
                if not isinstance(data, list):
                    detail = "unexpected payload shape"
                    break
                return data
            detail = f"HTTP {resp.status_code}"
            if resp.status_code == 404:
                break
            if attempt == 1:
                time.sleep(self.retry_cooldown)
        raise OracleError(f"Oracle request failed: {path} ({detail}). "
                          "Raising loudly — an empty or partial truth table "
                          "must never look like data.")

    def citation_count(self, doi: str) -> int:
        """The API's own count of incoming citations for one DOI."""
        d = canonical_doi(doi)
        if not d:
            raise OracleError(f"Not a DOI: {doi!r}")
        data = self._get(f"citation-count/doi:{d}")
        try:
            return int(data[0]["count"])
        except (IndexError, KeyError, TypeError, ValueError):
            raise OracleError(f"citation-count for {d} has an unexpected "
                              f"shape: {str(data)[:200]}")

    def citers(self, doi: str, cutoff_date: str) -> tuple[list[dict], dict]:
        """All DOI-registered works citing `doi`, created <= cutoff.
        -> (rows, report). rows: {key, doi, year}; report: how many raw
        rows fell out and why (written next to the frozen truth)."""
        m = _CUTOFF.match(cutoff_date or "")
        if not m:
            raise OracleError(f"cutoff must be a 31 December (YYYY-12-31), "
                              f"got {cutoff_date!r}: creation dates have "
                              "mixed precision and only a year-end cutoff "
                              "is exact for all of them.")
        year_max = int(m.group(1))
        d = canonical_doi(doi)
        if not d:
            raise OracleError(f"Not a DOI: {doi!r}")
        raw = self._get(f"citations/doi:{d}")
        expected = self.citation_count(d)
        if len(raw) != expected:
            raise OracleError(
                f"SELF-VALIDATION FAILED for {d}: the citations list has "
                f"{len(raw)} rows but /citation-count says {expected}. "
                "Refusing to write truth.")
        rep = {"raw_rows": len(raw), "no_doi": 0, "no_date": 0,
               "after_cutoff": 0, "duplicate": 0, "multi_doi": 0}
        seen, rows = set(), []
        for r in raw:
            dois = citing_dois(r.get("citing"))
            if not dois:
                rep["no_doi"] += 1                # outside the DOI universe
                continue
            creation = str(r.get("creation") or "")
            if not re.match(r"^\d{4}", creation):
                rep["no_date"] += 1               # cannot be placed in time
                continue
            if int(creation[:4]) > year_max:
                rep["after_cutoff"] += 1
                continue
            if len(dois) > 1:
                rep["multi_doi"] += 1             # keyed on its first DOI
            key = cite_key_from_parts(dois[0], "")
            if key in seen:
                rep["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"key": key, "doi": dois[0],
                         "year": int(creation[:4])})
        rep["truth_rows"] = len(rows)
        return rows, rep
