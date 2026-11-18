"""Select the Exp 2-C cohort BY RULE (PREREGISTRATION_EXP2C §4) — $0.

The rule, applied mechanically (no hand picking):
  1. CANDIDATE POOL: OpenAlex works that are research articles, published
     <= 2021, in the life/health sciences, with a DOI and a PMID, and a
     lifetime citation count in a wide window around the band.
     To spend oracle calls only where they can matter, the pool keeps a
     work only if OpenAlex's own year-by-year counts put its citations at
     the cutoff within a GENEROUS window around the band (0.8x to 1.25x).
     This is a speed filter: the oracle measurement in step 3 decides.
  2. Shuffle the pool with a FIXED seed (reproducible, order-independent
     of anything we know about the papers).
  3. Walk the shuffled list; for each candidate measure
       - its ORACLE truth size at the cutoff (OpenCitations, DOI-registered
         citers created <= cutoff — the same function the truth build uses),
       - how many of those citers each enumerator lists (index totals).
     ACCEPT when the truth size is inside the band AND Europe PMC lists at
     least --min-coverage of what OpenAlex lists (otherwise most records
     would be singletons and the certificate could never fire).
  4. Stop at --n accepted papers. The pilot paper is excluded.

The band comes from webaggs/design.py for the chosen eps and tau_max,
made robust to the uncertainty in the singleton share u (the probe's
estimate is rough), and converted from N (distinct records the agent
finds) to truth size with the probe's measured N/truth ratio.

Writes cohort/papers.txt ('Title | DOI | PMID', the truth-build input)
and cohort/selection_report.csv (EVERY candidate examined, accepted or
not, with the reason — the audit trail for the rule).

Usage:
    $env:RUN_ONLINE="1"
    python scripts/select_cohort.py --eps 0.7 --tau-max 1000 --cutoff 2023-12-31
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
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

REPO = Path(__file__).resolve().parents[1]
PILOT_DOIS = {"10.1038/s41586-021-03819-2"}      # AlphaFold 2: never in cohort
SEED = 20261117


def truth_band(eps, tau_max, page_size, u_lo, u_hi, n_over_truth, margin):
    """The truth-size window in which the certificate can fire for EVERY
    singleton share u in [u_lo, u_hi]: N must clear the psi requirement at
    the worst u (u_hi) and both walks must fit inside tau_max at the most
    page-hungry u (u_lo)."""
    lo_band = design.feasible_band(eps, page_size, tau_max, config.BETA,
                                   config.DELTA_M, u_hi)
    hi_band = design.feasible_band(eps, page_size, tau_max, config.BETA,
                                   config.DELTA_M, u_lo)
    if not lo_band or not hi_band or lo_band[0] >= hi_band[1]:
        raise SystemExit(f"No band robust to u in [{u_lo}, {u_hi}] at "
                         f"eps={eps}, tau_max={tau_max} — raise tau_max.")
    n_lo, n_hi = lo_band[0], hi_band[1]
    # N (what the agent finds) -> truth size, then shrink by a safety margin
    return (n_lo / n_over_truth * (1 + margin),
            n_hi / n_over_truth * (1 - margin), n_lo, n_hi)


def candidate_pool(oa, year_max, c_lo, c_hi, max_pool, cutoff_year):
    """OpenAlex works matching the pool rule, as dicts. Cursor-paged.
    Each carries `est_at_cutoff`: lifetime citations minus the citations
    OpenAlex attributes to years after the cutoff."""
    base = ["type:article", f"publication_year:<{year_max + 1}",
            "has_doi:true", "has_pmid:true",
            f"cited_by_count:>{c_lo}", f"cited_by_count:<{c_hi}"]
    out, note = [], ""
    # life sciences (1) or health sciences (4); fall back to no domain
    # filter if the API rejects it — the PMID requirement still applies
    for extra in (["primary_topic.domain.id:1|4"], []):
        out, cursor = [], "*"
        while cursor and len(out) < max_pool:
            status, data = oa._get(oa.BASE, {
                "filter": ",".join(base + extra),
                "select": "id,doi,display_name,publication_year,"
                          "cited_by_count,counts_by_year,ids",
                "per-page": 200, "cursor": cursor,
                **oa._key()})
            if status != 200 or data is None:
                out = None
                break
            for w in data.get("results") or []:
                pmid = str((w.get("ids") or {}).get("pmid") or "")
                pmid = pmid.rstrip("/").rsplit("/", 1)[-1]
                doi = canonical_doi(w.get("doi"))
                if doi and pmid.isdigit():
                    # plain-text title: drop markup like <i>...</i>
                    title = re.sub(r"<[^>]+>", "", str(
                        w.get("display_name") or ""))
                    after = sum(int(y.get("cited_by_count") or 0)
                                for y in w.get("counts_by_year") or []
                                if int(y.get("year") or 0) > cutoff_year)
                    life = int(w.get("cited_by_count") or 0)
                    out.append({"title": " ".join(title.split()),
                                "doi": doi, "pmid": pmid,
                                "year": w.get("publication_year"),
                                "lifetime": life,
                                "est_at_cutoff": life - after})
            cursor = (data.get("meta") or {}).get("next_cursor") \
                if data.get("results") else None
        if out:
            note = "domain filter: " + (extra[0] if extra else "none (fallback)")
            break
    if not out:
        raise SystemExit("OpenAlex returned no candidate pool.")
    return out, note


def index_total(ix, target, cutoff):
    """How many citers <= cutoff the index itself reports (one small call)."""
    try:
        handle = ix.resolve(target)
    except Exception:
        return None
    _src, page = ix.fetch_page(target, handle, ix.first_cursor, 1, cutoff,
                               "select")
    return page.total if page is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--tau-max", type=int, required=True)
    ap.add_argument("--cutoff", required=True, help="YYYY-12-31")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--page-size", type=int, default=config.INDEX_PAGE_SIZE)
    ap.add_argument("--u-lo", type=float, default=0.20)
    ap.add_argument("--u-hi", type=float, default=0.36)
    ap.add_argument("--n-over-truth", type=float, default=1.08,
                    help="distinct records found / truth size (probe: 1.08)")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-coverage", type=float, default=0.75,
                    help="Europe PMC total / OpenAlex total, at the cutoff")
    ap.add_argument("--year-max", type=int, default=2021)
    ap.add_argument("--max-checked", type=int, default=150)
    ap.add_argument("--time-budget", type=float, default=780.0,
                    help="seconds; stop examining and write what exists "
                         "(keeps a CI run inside its time limit)")
    args = ap.parse_args()
    if os.environ.get("RUN_ONLINE") != "1":
        raise SystemExit("RUN_ONLINE != 1 — network access is a deliberate "
                         "per-session decision.")

    t_lo, t_hi, n_lo, n_hi = truth_band(
        args.eps, args.tau_max, args.page_size, args.u_lo, args.u_hi,
        args.n_over_truth, args.margin)
    print(f"band: eps={args.eps}, tau_max={args.tau_max}, robust to u in "
          f"[{args.u_lo}, {args.u_hi}] -> N in [{n_lo:,.0f}, {n_hi:,.0f}] "
          f"-> truth size in [{t_lo:,.0f}, {t_hi:,.0f}]")

    oa, ep, oracle = OpenAlexIndex(), EuropePMCIndex(), OpenCitationsDriver()
    # lifetime counts run ahead of the count at the cutoff, so the pool
    # window is wide; the ORACLE measurement below is what decides
    cutoff_year = int(args.cutoff[:4])
    pool, note = candidate_pool(oa, args.year_max, int(t_lo * 0.8),
                                int(t_hi * 3.0), max_pool=4000,
                                cutoff_year=cutoff_year)
    n_pool = len(pool)
    # speed filter on OpenAlex's own yearly counts (the oracle decides)
    pool = [c for c in pool
            if 0.8 * t_lo <= c["est_at_cutoff"] <= 1.25 * t_hi]
    print(f"pool: {n_pool} works in the lifetime window, {len(pool)} with "
          f"an estimated count at the cutoff near the band")
    pool = [c for c in pool if c["doi"] not in PILOT_DOIS]
    pool.sort(key=lambda c: c["doi"])              # order-independent start
    random.Random(SEED).shuffle(pool)
    print(f"candidate pool: {len(pool)} works ({note}); seed {SEED}")

    report, accepted = [], []
    t_start = time.time()
    for c in pool:
        if len(accepted) >= args.n or len(report) >= args.max_checked:
            break
        if time.time() - t_start > args.time_budget:
            print("  time budget reached — writing what was examined")
            break
        row = dict(c, truth="", openalex_total="", europepmc_total="",
                   coverage="", decision="", why="")
        try:
            rows, _rep = oracle.citers(c["doi"], args.cutoff)
        except OracleError as e:
            row.update(decision="skip", why=f"oracle: {str(e)[:80]}")
            report.append(row)
            continue
        row["truth"] = len(rows)
        if not (t_lo <= len(rows) <= t_hi):
            row.update(decision="reject", why="truth size outside the band")
            report.append(row)
            continue
        tgt = Target(title=c["title"], doi=c["doi"], pmid=c["pmid"])
        n_oa = index_total(oa, tgt, args.cutoff)
        n_ep = index_total(ep, tgt, args.cutoff)
        row.update(openalex_total=n_oa, europepmc_total=n_ep)
        if not n_oa or not n_ep:
            row.update(decision="reject", why="an enumerator cannot list it")
        else:
            cov = n_ep / n_oa
            row["coverage"] = f"{cov:.2f}"
            if cov < args.min_coverage:
                row.update(decision="reject",
                           why=f"Europe PMC coverage < {args.min_coverage}")
            else:
                row.update(decision="ACCEPT", why="")
                accepted.append(c)
        report.append(row)
        print(f"  {row['decision']:6s} truth={row['truth']:>6} "
              f"cov={row['coverage'] or '-':>4}  {c['title'][:60]}")

    out_dir = REPO / "cohort"
    out_dir.mkdir(exist_ok=True)
    if not report:
        raise SystemExit("No candidate was examined — the pool is empty "
                         "near this band; widen the band or tau_max.")
    with open(out_dir / "selection_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    lines = [
        "# Exp 2-C cohort papers — selected BY RULE by scripts/select_cohort.py",
        f"# rule: eps={args.eps}, tau_max={args.tau_max}, cutoff={args.cutoff}, "
        f"truth size in [{t_lo:,.0f}, {t_hi:,.0f}] (robust to u in "
        f"[{args.u_lo}, {args.u_hi}]), Europe PMC coverage >= "
        f"{args.min_coverage}, year <= {args.year_max}, seed {SEED}",
        "# audit trail (every candidate examined): cohort/selection_report.csv",
        "# STAGED, not frozen: frozen only by build_citation_truth.py after "
        "the supervisor signs.",
        "# Title | DOI | PMID",
    ] + [f"{c['title'].replace('|', '-')} | {c['doi']} | {c['pmid']}"
         for c in accepted]
    (out_dir / "papers.txt").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"\naccepted {len(accepted)}/{args.n} after examining "
          f"{len(report)} candidates -> cohort/papers.txt")
    if len(accepted) < args.n:
        print("  fewer than requested: widen --max-checked or the band.")


if __name__ == "__main__":
    main()
