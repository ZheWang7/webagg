"""Exp 2-C enumerator PROBE ($0, no keys): measure the two candidate
citation indexes BEFORE the pagination plug-in is built (prereg §6.2/§9).

Four questions, each of which sets a prereg parameter:
  1. page size     — is the `limit`/`pageSize` we request honored?
                     (it becomes Y_CAP, so it must be a real bound)
  2. depth cap     — how deep can pagination actually walk? Semantic
                     Scholar's citations endpoint is believed to cap
                     offset around 10,000 — if true, Y_CAP and the
                     cohort band must shrink to fit under it.
  3. keyability    — what fraction of entries carry a DOI (the primary
                     record key)?
  4. overlap       — do the two indexes cover the same DOIs (double
                     coverage needs real overlap)?

Probes BOTH indexes on the SAME paper (default: AlphaFold, the kind of
biomedical giant the cohort band targets) so overlap is meaningful.

Usage:
    $env:RUN_ONLINE="1"
    python scripts/probe_enumerators.py
    python scripts/probe_enumerators.py --pmid 32015508 --doi 10.1038/s41586-020-2008-3
"""
from __future__ import annotations

import argparse
import os
import time

import requests

# AlphaFold 2 (Jumper et al. 2021): same paper known to both indexes
DEFAULT_PMID = "34265844"
DEFAULT_DOI = "10.1038/s41586-021-03819-2"

S2 = "https://api.semanticscholar.org/graph/v1/paper"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
PAUSE = 1.5          # polite pacing: both APIs are free/anonymous


def get(url, params):
    """One GET with loud, informative failure (probe = diagnosis tool)."""
    time.sleep(PAUSE)
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        # the error BODY is the finding (e.g. S2 names its offset cap)
        return {"__status": r.status_code, "__body": r.text[:300]}
    return r.json()


def s2_doi(entry):
    ids = (entry.get("citingPaper") or {}).get("externalIds") or {}
    return (ids.get("DOI") or "").lower() or None


def epmc_doi(entry):
    return (entry.get("doi") or "").lower() or None


def probe_s2(doi):
    print(f"\n=== Semantic Scholar: {S2}/DOI:{doi}/citations ===")
    url = f"{S2}/DOI:{doi}/citations"
    meta = get(f"{S2}/DOI:{doi}", {"fields": "citationCount,title"})
    total = meta.get("citationCount")
    print(f"paper: {meta.get('title')!r}, citationCount={total}")

    # Q1: are requested page sizes honored? (each becomes a candidate Y_CAP)
    for lim in (25, 100, 500):
        j = get(url, {"fields": "externalIds,title", "limit": lim, "offset": 0})
        got = len(j.get("data", [])) if "data" in j else j
        print(f"  limit={lim:4d} -> entries on page: {got}")

    # Q2: the depth cap — walk the offset up until the API says no
    cap = None
    for off in (1000, 5000, 9000, 9900, 9975, 10000, 12000):
        j = get(url, {"fields": "externalIds", "limit": 25, "offset": off})
        if "data" in j:
            print(f"  offset={off:5d} -> ok ({len(j['data'])} entries, "
                  f"next={j.get('next')})")
        else:
            print(f"  offset={off:5d} -> HTTP {j['__status']}: {j['__body']}")
            cap = off
            break
    print(f"  DEPTH CAP: {'~'+str(cap) if cap else 'none found <= 12000'}")

    # Q3: DOI keyability on a sample page
    j = get(url, {"fields": "externalIds,title", "limit": 100, "offset": 0})
    rows = j.get("data", [])
    dois = [d for d in (s2_doi(e) for e in rows) if d]
    print(f"  keyability: {len(dois)}/{len(rows)} entries carry a DOI")
    return total, set(dois), cap


def probe_epmc(pmid):
    print(f"\n=== Europe PMC: {EPMC}/MED/{pmid}/citations ===")
    url = f"{EPMC}/MED/{pmid}/citations"
    j = get(url, {"page": 1, "pageSize": 100, "format": "json"})
    if "__status" in j:
        print(f"  FAILED: HTTP {j['__status']}: {j['__body']}")
        return None, set(), None
    total = j.get("hitCount")
    print(f"  hitCount={total}")

    # Q1: page sizes honored?
    for ps in (25, 100, 1000):
        jj = get(url, {"page": 1, "pageSize": ps, "format": "json"})
        rows = (jj.get("citationList") or {}).get("citation", [])
        print(f"  pageSize={ps:4d} -> entries on page: {len(rows)}")

    # Q2: depth — can page*pageSize walk past 10k? past the full count?
    cap = None
    for page in (10, 50, 101, 150, 250):          # x100 = position 1k..25k
        jj = get(url, {"page": page, "pageSize": 100, "format": "json"})
        rows = (jj.get("citationList") or {}).get("citation", []) \
            if "__status" not in jj else None
        if rows is None:
            print(f"  page={page:4d} (pos {page*100}) -> "
                  f"HTTP {jj['__status']}: {jj['__body']}")
            cap = page * 100
            break
        print(f"  page={page:4d} (pos {page*100}) -> {len(rows)} entries")
        if not rows:
            break                                  # walked past the end: fine
    print(f"  DEPTH CAP: {'~'+str(cap) if cap else 'none hit'}")

    # Q3: DOI keyability
    jj = get(url, {"page": 1, "pageSize": 100, "format": "json"})
    rows = (jj.get("citationList") or {}).get("citation", [])
    dois = [d for d in (epmc_doi(e) for e in rows) if d]
    print(f"  keyability: {len(dois)}/{len(rows)} entries carry a DOI")
    return total, set(dois), cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmid", default=DEFAULT_PMID)
    ap.add_argument("--doi", default=DEFAULT_DOI)
    args = ap.parse_args()
    # same opt-in gate as the online sanity tests (shell-only, never .env)
    if os.environ.get("RUN_ONLINE") != "1":
        raise SystemExit("RUN_ONLINE != 1 — network probes are a deliberate "
                         "per-session decision ($0 here, but the rule is the rule).")

    s2_total, s2_dois, s2_cap = probe_s2(args.doi)
    ep_total, ep_dois, ep_cap = probe_epmc(args.pmid)

    # Q4: overlap on the sampled first pages (same paper, both indexes)
    print("\n=== Verdict inputs ===")
    print(f"totals: S2={s2_total}  EPMC={ep_total}  "
          f"(vs OpenAlex truth at grading time)")
    if s2_dois and ep_dois:
        inter = len(s2_dois & ep_dois)
        print(f"first-page DOI overlap: {inter}/{min(len(s2_dois), len(ep_dois))} "
              "(pages are differently ordered — any overlap > 0 confirms "
              "the indexes key-join; the real overlap is measured in PILOT 2)")
    print("depth caps: S2", s2_cap, "| EPMC", ep_cap)
    print("\n-> paste this whole output into the chat; Y_CAP (page size), "
          "the live eps and the cohort citer band are chosen from it "
          "(prereg section 6.2/6.3).")


if __name__ == "__main__":
    main()
