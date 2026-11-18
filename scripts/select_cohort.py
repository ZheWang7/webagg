"""Select the Exp 2-C cohort BY RULE (PREREGISTRATION_EXP2C §4) — $0.

The rule, applied mechanically (no hand picking):
  1. CANDIDATE POOL: OpenAlex works that are research articles, published
     <= 2021, in the life/health sciences, with a DOI and a PMID, whose
     citation count at the cutoff (lifetime minus OpenAlex's own counts
     for later years) is near the band.
  2. Shuffle the pool with a FIXED seed (reproducible, order-independent
     of anything we know about the papers).
  3. Walk the shuffled list. For each candidate MEASURE, from the two
     enumerators only:
       - how many citers <= cutoff each index lists (n_OA, n_EP),
       - their overlap, from a sample of each index looked up in the other,
     giving the number of distinct records the agent will find (N) and
     the share only one index lists (the singleton share u).
     ACCEPT when the certificate's own precondition holds with a pad:
       (u + pad) + psi(N) < eps          psi small enough
       pages for both walks + overhead <= tau_max   (at u - pad)
     and the oracle really has the paper (truth size at the cutoff is at
     least --min-oracle-ratio of what OpenAlex lists).
  4. Stop at --n accepted papers. The pilot paper is excluded.

Why N and u, not the truth size: the band of webaggs/design.py is about
what the AGENT finds. The oracle's list is usually smaller than
OpenAlex's (median 17% in the first selection run), so a truth-size
window rejected papers the agent could certify. The oracle is used here
only as a sanity check and to record the truth size; no recall against
the oracle is computed at selection time.

Writes cohort/papers.txt ('Title | DOI | PMID', the truth-build input)
and cohort/selection_report.csv (EVERY candidate examined, accepted or
not, with the reason — the audit trail for the rule).

Usage:
    $env:RUN_ONLINE="1"
    python scripts/select_cohort.py --eps 0.6 --tau-max 1150 --cutoff 2023-12-31
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_enumerators as probe                # noqa: E402  (membership helpers)

REPO = Path(__file__).resolve().parents[1]
OVERHEAD = 20                                    # non-index search occasions


class EnumeratorRefused(RuntimeError):
    """An index refused access (missing/exhausted key or quota). Nothing
    can be measured, so the selection must stop — not skip every paper."""


def fits(n, u, eps, tau_max, page_size, pad):
    """The certificate's precondition for a paper with N=n distinct records
    and singleton share u, robust to an error of +-pad in u:
    -> (ok, why). Uses the same formulas as the live stop test."""
    psi = design.psi_at(int(n), page_size, tau_max, config.BETA,
                        config.DELTA_M)
    if (u + pad) + psi >= eps:
        return False, f"u+pad+psi = {u + pad + psi:.2f} >= eps"
    pages = design.pages_for(n, page_size, max(u - pad, 0.0)) + OVERHEAD
    if pages > tau_max:
        return False, f"walks need {pages:.0f} occasions > tau_max"
    return True, ""


def measure(oa, ep, cand, cutoff, sample):
    """-> dict(n_oa, n_ep, a, b, n, u) from the two enumerators, or a
    string saying which one failed. a = share of OpenAlex's citers that
    Europe PMC also lists; b = the reverse."""
    tgt = Target(title=cand["title"], doi=cand["doi"], pmid=cand["pmid"])
    try:
        # the pool already carries the OpenAlex id; Europe PMC uses the PMID
        h_oa = cand.get("oa_id") or oa.resolve(tgt)
        h_ep = ep.resolve(tgt)
    except Exception as e:
        return f"cannot resolve: {str(e)[:60]}"
    samples, totals = {}, {}
    for ix, h in ((oa, h_oa), (ep, h_ep)):
        got, cur = [], ix.first_cursor
        while cur and len(got) < sample:
            src, page = ix.fetch_page(tgt, h, cur, 50, cutoff, "select")
            if page is None and src.status == "paywall":
                raise EnumeratorRefused(
                    f"{ix.name} refused access (HTTP 401/402/403). For "
                    "OpenAlex: set OPENALEX_API_KEY in .env, or the key's "
                    "daily free usage is spent — re-run tomorrow.")
            if page is None:
                return f"{ix.name} page failed ({src.status})"
            totals.setdefault(ix.name, page.total)
            got += probe.page_dois(page)
            cur = page.next_cursor
        samples[ix.name] = got[:sample]
    n_oa, n_ep = totals.get("openalex"), totals.get("europepmc")
    if not n_oa or not n_ep or not samples["openalex"] \
            or not samples["europepmc"]:
        return "an enumerator lists nothing"
    a = len(probe.in_europepmc(ep, h_ep, samples["openalex"])) \
        / len(samples["openalex"])
    b = len(probe.in_openalex(oa, h_oa, samples["europepmc"])) \
        / len(samples["europepmc"])
    inter = (a * n_oa + b * n_ep) / 2            # two estimates, averaged
    n = n_oa + n_ep - inter
    return dict(n_oa=n_oa, n_ep=n_ep, a=a, b=b, n=n, u=(n - inter) / n)
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
                                "oa_id": str(w.get("id") or "").rsplit(
                                    "/", 1)[-1],
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
    ap.add_argument("--pad", type=float, default=0.03,
                    help="safety pad on the measured singleton share u")
    ap.add_argument("--sample", type=int, default=100,
                    help="citers sampled from each index for the overlap")
    ap.add_argument("--min-oracle-ratio", type=float, default=0.6,
                    help="oracle truth size / OpenAlex total, at the cutoff")
    ap.add_argument("--year-max", type=int, default=2021)
    ap.add_argument("--max-checked", type=int, default=150)
    ap.add_argument("--time-budget", type=float, default=780.0,
                    help="seconds; stop examining and write what exists "
                         "(keeps a CI run inside its time limit)")
    args = ap.parse_args()
    if os.environ.get("RUN_ONLINE") != "1":
        raise SystemExit("RUN_ONLINE != 1 — network access is a deliberate "
                         "per-session decision.")

    # a generous N window (u from 0.20 to 0.45): it only sizes the pool —
    # below it psi cannot clear at any plausible u, above it the walks
    # cannot fit; `fits` decides each candidate on its own measurements
    n_lo = design.n_needed(args.eps, args.page_size, args.tau_max,
                           config.BETA, config.DELTA_M, u=0.20 + args.pad)
    n_hi = args.page_size * (args.tau_max - OVERHEAD) / (2 - 0.45)
    print(f"rule: eps={args.eps}, tau_max={args.tau_max}, pad={args.pad}; "
          f"pool sized for N in [{n_lo:,.0f}, {n_hi:,.0f}]")

    oa, ep, oracle = OpenAlexIndex(), EuropePMCIndex(), OpenCitationsDriver()
    cutoff_year = int(args.cutoff[:4])
    pool, note = candidate_pool(oa, args.year_max, int(n_lo * 0.9),
                                int(n_hi * 3.0), max_pool=4000,
                                cutoff_year=cutoff_year)
    n_pool = len(pool)
    # N is a little above OpenAlex's own total, so keep works whose
    # OpenAlex count at the cutoff could give an N inside the window
    pool = [c for c in pool
            if 0.9 * n_lo <= c["est_at_cutoff"] <= 1.0 * n_hi
            and c["doi"] not in PILOT_DOIS]
    pool.sort(key=lambda c: c["doi"])              # order-independent start
    random.Random(SEED).shuffle(pool)
    print(f"candidate pool: {n_pool} works in the lifetime window, "
          f"{len(pool)} near the band ({note}); seed {SEED}")

    report, accepted = [], []
    t_start = time.time()
    for c in pool:
        if len(accepted) >= args.n or len(report) >= args.max_checked:
            break
        if time.time() - t_start > args.time_budget:
            print("  time budget reached — writing what was examined")
            break
        row = dict(c, openalex_total="", europepmc_total="", a="", b="",
                   n_est="", u_est="", truth="", decision="", why="")
        try:
            m = measure(oa, ep, c, args.cutoff, args.sample)
        except EnumeratorRefused as e:
            # stop here: papers.txt is NOT rewritten with a partial cohort
            raise SystemExit(f"STOPPED after {len(report)} candidates, "
                             f"{len(accepted)} accepted so far: {e}")
        if isinstance(m, str):
            row.update(decision="skip", why=m)
        else:
            row.update(openalex_total=m["n_oa"], europepmc_total=m["n_ep"],
                       a=f"{m['a']:.2f}", b=f"{m['b']:.2f}",
                       n_est=f"{m['n']:.0f}", u_est=f"{m['u']:.3f}")
            ok, why = fits(m["n"], m["u"], args.eps, args.tau_max,
                           args.page_size, args.pad)
            if not ok:
                row.update(decision="reject", why=why)
            else:
                # the oracle is consulted only now, as a sanity check
                try:
                    rows, _rep = oracle.citers(c["doi"], args.cutoff)
                except OracleError as e:
                    row.update(decision="skip", why=f"oracle: {str(e)[:70]}")
                else:
                    row["truth"] = len(rows)
                    if len(rows) < args.min_oracle_ratio * m["n_oa"]:
                        row.update(decision="reject",
                                   why="oracle lists too few of its citers")
                    else:
                        row.update(decision="ACCEPT", why="")
                        accepted.append(c)
        report.append(row)
        print(f"  {row['decision']:6s} N={row['n_est'] or '-':>6} "
              f"u={row['u_est'] or '-':>5} truth={row['truth'] or '-':>6}  "
              f"{c['title'][:50]}  {row['why']}")

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
        f"accept iff (u + {args.pad}) + psi(N) < eps and both walks fit "
        f"in tau_max, with N and u measured from the two enumerators; "
        f"year <= {args.year_max}, seed {SEED}",
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
