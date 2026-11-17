"""Exp 2-C PROBE ($0): measure the oracle and the two enumerators on one
paper BEFORE any live spend (PREREGISTRATION_EXP2C rev 3 §6.3).

It answers, with numbers, the questions the design cannot assume:
  1. ORACLE     — does OpenCitations return the full citing list, does it
                  self-validate against its own count, and how big is the
                  DOI-registered truth at the cutoff?
  2. ENUMERATORS— do OpenAlex and Europe PMC resolve the paper, honor the
                  page size we request (it becomes Y_CAP), and return DOIs?
  3. REACH      — of a random sample of TRUTH records, what share can the
                  agent reach at all (in at least one enumerator)? That is
                  the recall ceiling.
  4. OVERLAP    — what share of what the agent finds will be in only ONE
                  enumerator (singletons, u)? This decides whether the
                  certificate can fire: it needs  u + psi < eps.
  5. THE BAND   — given the measured u: the smallest tau_max and the range
                  of N for which the certificate is reachable, per eps.

Default paper: AlphaFold 2 (a biomedical giant, the cohort's kind).

Usage:
    $env:RUN_ONLINE="1"
    python scripts/probe_enumerators.py
    python scripts/probe_enumerators.py --doi 10.1038/s41586-020-2649-2 --pmid 32939066
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config, design              # noqa: E402
from webaggs.canon import canonical_doi         # noqa: E402
from webaggs.indexes import (EuropePMCIndex, OpenAlexIndex,  # noqa: E402
                             Target)
from webaggs.oracle_opencitations import (OpenCitationsDriver,  # noqa: E402
                                          OracleError)

DEFAULT_TITLE = "Highly accurate protein structure prediction with AlphaFold"
DEFAULT_DOI = "10.1038/s41586-021-03819-2"
DEFAULT_PMID = "34265844"
OUT = config.DATA_DIR / "ci" / "exp2c_probe.md"


ALIASED: list[str] = []    # OpenAlex answers under a different primary DOI


def page_dois(page) -> list[str]:
    return [d for d in (canonical_doi(r.get("doi")) for r in page.rows) if d]


def in_openalex(oa, handle, dois) -> set:
    """Which of `dois` cite the target according to OpenAlex (batched OR)."""
    found = set()
    for i in range(0, len(dois), 40):
        chunk = dois[i:i + 40]
        status, data = oa._get(oa.BASE, {
            "filter": f"cites:{handle},doi:" + "|".join(
                f"https://doi.org/{d}" for d in chunk),
            "select": "doi", "per-page": 100, **oa._key()})
        for w in (data or {}).get("results") or []:
            d = canonical_doi(w.get("doi"))
            # count a hit only when OpenAlex lists the work under the SAME
            # DOI we asked for: if it answers with a different primary DOI
            # (preprint vs published version) the agent's key will not join
            # the truth key, so it is NOT reachable for grading
            if d and d in chunk:
                found.add(d)
            elif d:
                ALIASED.append(d)
    return found


def in_europepmc(ep, handle, dois) -> set:
    """Which of `dois` cite the target according to Europe PMC (one small
    query per DOI — the search syntax has no reliable batch OR on DOI)."""
    found = set()
    for d in dois:
        status, data = ep._get(ep.BASE, {
            "query": f'CITES:{handle}_MED AND DOI:"{d}"',
            "resultType": "idlist", "pageSize": 1, "format": "json"})
        if (data or {}).get("hitCount"):
            found.add(d)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default=DEFAULT_TITLE)
    ap.add_argument("--doi", default=DEFAULT_DOI)
    ap.add_argument("--pmid", default=DEFAULT_PMID)
    ap.add_argument("--cutoff", default="2023-12-31")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--page-size", type=int, default=config.INDEX_PAGE_SIZE)
    args = ap.parse_args()
    # same opt-in gate as the online sanity tests (shell-only, never .env)
    if os.environ.get("RUN_ONLINE") != "1":
        raise SystemExit("RUN_ONLINE != 1 — network probes are a deliberate "
                         "per-session decision ($0 here, but the rule is the rule).")

    lines: list[str] = []

    def say(text=""):
        print(text)
        lines.append(text)

    target = Target(title=args.title, doi=args.doi, pmid=args.pmid)
    rng = random.Random(20261117)                 # fixed seed: reproducible
    say(f"# Exp 2-C probe — {args.title}")
    say(f"DOI {args.doi} · PMID {args.pmid} · cutoff {args.cutoff} · "
        f"page size {args.page_size}")

    # --- 1. the oracle ----------------------------------------------------
    say("\n## 1. Oracle: OpenCitations")
    truth = []
    try:
        t0 = time.time()
        rows, rep = OpenCitationsDriver().citers(args.doi, args.cutoff)
        truth = [r["doi"] for r in rows]
        say(f"- full list fetched and SELF-VALIDATED against /citation-count "
            f"in {time.time() - t0:.0f}s")
        say(f"- raw rows {rep['raw_rows']}: no DOI {rep['no_doi']}, undated "
            f"{rep['no_date']}, after cutoff {rep['after_cutoff']}, "
            f"duplicates {rep['duplicate']}")
        say(f"- **truth size at cutoff: {len(truth)} DOI-registered citers**")
    except OracleError as e:
        say(f"- ORACLE FAILED: {e}")

    # --- 2. the enumerators ----------------------------------------------
    say("\n## 2. Enumerators")
    oa, ep = OpenAlexIndex(), EuropePMCIndex()
    info = {}
    for ix in (oa, ep):
        try:
            handle = ix.resolve(target)
        except Exception as e:
            say(f"- {ix.name}: CANNOT RESOLVE — {e}")
            continue
        src, page = ix.fetch_page(target, handle, ix.first_cursor,
                                  args.page_size, args.cutoff, "probe")
        if page is None:
            say(f"- {ix.name}: first page FAILED (status {src.status})")
            continue
        dois = page_dois(page)
        say(f"- {ix.name}: handle {handle}, index total {page.total}, first "
            f"page {len(page.rows)} rows (asked {args.page_size}), "
            f"{len(dois)} with DOI, next cursor "
            f"{'yes' if page.next_cursor else 'NO'}")
        # walk two more pages to sample beyond the head of the list
        sample, cur, p = list(dois), page.next_cursor, page
        for _ in range(2):
            if not cur:
                break
            _s, p = ix.fetch_page(target, handle, cur, args.page_size,
                                  args.cutoff, "probe")
            if p is None:
                break
            sample += page_dois(p)
            cur = p.next_cursor
        info[ix.name] = dict(handle=handle, total=page.total or 0,
                             sample=sample)
    if os.environ.get("OPENALEX_API_KEY") is None:
        say("- NOTE: OPENALEX_API_KEY not set — OpenAlex anonymous access is "
            "metered per IP; set the free key before a full walk")

    if len(info) < 2 or not truth:
        say("\n**Probe incomplete — fix the failures above before going on.**")
    else:
        H_oa, H_ep = info["openalex"]["handle"], info["europepmc"]["handle"]
        # --- 3. reach: a random sample of TRUTH, looked up in both -------
        say("\n## 3. Reach (recall ceiling against the oracle)")
        samp = rng.sample(truth, min(args.sample, len(truth)))
        t_oa = in_openalex(oa, H_oa, samp)
        t_ep = in_europepmc(ep, H_ep, samp)
        either, both = t_oa | t_ep, t_oa & t_ep
        n = len(samp)
        say(f"- sample of {n} truth records: in OpenAlex {len(t_oa)/n:.0%}, "
            f"in Europe PMC {len(t_ep)/n:.0%}, in BOTH {len(both)/n:.0%}, "
            f"in NEITHER {1 - len(either)/n:.0%}")
        ceiling = len(either) / n
        say(f"- DOI aliasing: for {len(ALIASED)} of the {n} sampled truth "
            "DOIs OpenAlex lists the citing work under a DIFFERENT primary "
            "DOI (counted as not reachable: the keys would not join)")
        say(f"- **recall ceiling ~ {ceiling:.0%}** (truth records reachable "
            "through at least one enumerator)")

        # --- 4. overlap: what the AGENT will see --------------------------
        say("\n## 4. Overlap between the enumerators (singleton share u)")
        s_oa = info["openalex"]["sample"][:args.sample]
        s_ep = info["europepmc"]["sample"][:args.sample]
        a = len(in_europepmc(ep, H_ep, s_oa)) / max(len(s_oa), 1)
        b = len(in_openalex(oa, H_oa, s_ep)) / max(len(s_ep), 1)
        n_oa, n_ep = info["openalex"]["total"], info["europepmc"]["total"]
        inter = (a * n_oa + b * n_ep) / 2          # two estimates, averaged
        union = n_oa + n_ep - inter
        u = (union - inter) / union if union else 1.0
        say(f"- of OpenAlex's citers, {a:.0%} are also in Europe PMC; of "
            f"Europe PMC's, {b:.0%} are also in OpenAlex")
        say(f"- estimated distinct records N ~ {union:,.0f}; "
            f"**singleton share u ~ {u:.2f}**")
        in_truth = len(set(s_oa) & set(truth)) / max(len(s_oa), 1)
        say(f"- of OpenAlex's sampled citers, {in_truth:.0%} are in the "
            "oracle's truth (the rest are spurious for grading, not harmful "
            "to recall)")

        # --- 5. the feasibility band --------------------------------------
        say("\n## 5. Can the certificate fire? (needs u + psi < eps)")
        say("| eps | min tau_max | N must be between | this paper's N | verdict |")
        say("|---|---|---|---|---|")
        for eps in (0.5, 0.6, 0.7, 0.8, 0.9):
            if u >= eps:
                say(f"| {eps} | — | — | {union:,.0f} | impossible: u >= eps |")
                continue
            tau = design.min_tau(eps, config.BETA, config.DELTA_M, u)
            lo, hi = design.feasible_band(eps, args.page_size, tau,
                                          config.BETA, config.DELTA_M, u)
            fits = "fits" if lo <= union <= hi else (
                "too few citers" if union < lo else "too many for this tau")
            say(f"| {eps} | {tau} | {lo:,.0f} – {hi:,.0f} | {union:,.0f} | "
                f"{fits} |")
        say("\n(min tau_max is the smallest budget with a 25%-wide band; a "
            "larger tau_max widens the band and moves it up.)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nreport written to {OUT} — paste it into the chat.")


if __name__ == "__main__":
    main()
