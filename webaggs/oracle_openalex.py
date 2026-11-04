"""The withheld oracle: OpenAlex (webaggr.pdf §9 role, citation domain).

Endpoint: GET https://api.openalex.org/works?filter=cites:{WORK_ID},
to_publication_date:{CUTOFF}  — the open scholarly graph, no key, no
bot-wall (CI-verified 2026-11-03). Truth is DEFINED by this API and DENIED
to the agent at run time (config.DENY_DOMAINS covers openalex.org). This
driver is used ONLY by truth-build / cohort tooling, never by the pipeline.

Scope rules it encodes (PREREGISTRATION_EXP2C.md):
  * records = works citing the target, published ON OR BEFORE the frozen
    cutoff date (the citation-domain analogue of "retired players only" —
    citation lists grow daily; the cutoff freezes them);
  * record key: DOI (lowercased, scheme-stripped) when OpenAlex has one,
    else "t:" + normalized title — the same keys canon.canonical_cite_key
    produces on the extraction side, so truth and assertions join exactly;
  * SELF-VALIDATION: the cursor walk must collect exactly meta.count rows
    (the API's own count for the filtered query), else the driver refuses
    to emit truth — a partially paginated truth table is structurally
    impossible.
"""
from __future__ import annotations

import os
import time

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout

from .canon import cite_key_from_parts

_BASE = "https://api.openalex.org/works"
_UA = {"User-Agent": "webagg-sports research client (Exp 2-C truth build)",
       "Accept": "application/json"}


class OracleError(RuntimeError):
    pass


class OpenAlexDriver:
    """Paced, retried once on transient errors, loud on everything else."""

    def __init__(self, pace_seconds: float = 0.5, client=None,
                 retry_cooldown: float = 10.0):
        self.pace = pace_seconds
        self.client = client or requests          # tests inject fakes
        self.retry_cooldown = retry_cooldown
        self._last = 0.0

    def _get(self, params: dict) -> dict:
        # OpenAlex meters anonymous access (shared $0.10/day per IP, per its
        # own meta notice); a free account key gets its own budget. Read it
        # from OPENALEX_API_KEY (via .env) when present.
        key = os.environ.get("OPENALEX_API_KEY")
        if key:
            params = {**params, "api_key": key}
        wait = self.pace - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        detail = "no response"
        for attempt in (1, 2):
            self._last = time.time()
            try:
                resp = self.client.get(_BASE, params=params, headers=_UA,
                                       timeout=(10, 60))
            except (ReqTimeout, ReqConnectionError) as e:
                detail = type(e).__name__
                if attempt == 1:
                    time.sleep(self.retry_cooldown)
                continue
            if resp.status_code == 200:
                return resp.json()
            detail = f"HTTP {resp.status_code}"
            if resp.status_code == 404:
                break                             # bad work id: not transient
            if attempt == 1:
                time.sleep(self.retry_cooldown)
        raise OracleError(f"Oracle request failed: {params} ({detail}). "
                          "Raising loudly — an empty or partial truth table "
                          "must never look like data.")

    def work_meta(self, work_id: str) -> dict:
        """Display name, year and total cited_by_count for one work."""
        data = self._get({"filter": f"ids.openalex:{work_id}", "per-page": 1})
        results = data.get("results") or []
        if not results:
            raise OracleError(f"Work {work_id!r} not found on OpenAlex — "
                              "check the W-id (from the work's openalex.org "
                              "URL).")
        w = results[0]
        return {"id": w.get("id"), "title": w.get("display_name"),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count")}

    def citers(self, work_id: str, cutoff_date: str) -> list[dict]:
        """All works citing `work_id` published <= cutoff_date (YYYY-MM-DD),
        one dict per citer: {key, doi, title, year}. Cursor-paginated and
        self-validated: collected rows must equal the API's meta.count."""
        rows, cursor, expected = [], "*", None
        flt = f"cites:{work_id},to_publication_date:{cutoff_date}"
        while cursor:
            data = self._get({"filter": flt, "per-page": 200,
                              "cursor": cursor})
            meta = data.get("meta") or {}
            if expected is None:
                expected = meta.get("count")
                if expected is None:
                    raise OracleError("meta.count missing — schema change; "
                                      "refusing to guess completeness.")
            for w in data.get("results") or []:
                title = w.get("display_name") or ""
                doi = w.get("doi") or ""
                key = cite_key_from_parts(doi, title)
                if key is None:
                    # a work with neither DOI nor usable title cannot be
                    # graded; refuse rather than silently shrink truth
                    raise OracleError(f"Citer {w.get('id')} has no DOI and "
                                      f"no usable title ({title!r}) — "
                                      "cannot form a grading key.")
                rows.append({"key": key, "doi": doi.lower(),
                             "title": title,
                             "year": w.get("publication_year")})
            cursor = meta.get("next_cursor")
        if len(rows) != expected:
            raise OracleError(
                f"SELF-VALIDATION FAILED for {work_id}: cursor walk "
                f"collected {len(rows)} citers but the API's meta.count says "
                f"{expected}. Pagination or filter is wrong — refusing to "
                "write truth.")
        # distinct keys: two citers may collide (same DOI reposted) — truth
        # must have one record per key, loudly deduplicated with a report
        seen, out, dupes = set(), [], 0
        for r in rows:
            if r["key"] in seen:
                dupes += 1
                continue
            seen.add(r["key"])
            out.append(r)
        if dupes:
            print(f"[note] {dupes} duplicate citation keys collapsed "
                  f"({len(out)} distinct of {len(rows)} rows)")
        return out
