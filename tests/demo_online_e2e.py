"""ONLINE end-to-end DEMO after ch. 12 (chapters 3-12, real APIs, figures).

Not a seam test (ch. 12 has no degradable seam) -- this is the PIPELINE-level
live check + supervisor demo: one budget-capped real run of end_to_end(),
then invariant checks and plain-language figures showing what the agent did
and why the result can be trusted:

    fig 1  the funnel        searches -> fetch attempts -> pages with text
                             -> judged relevant -> certified facts -> records
    fig 2  source classes    which KINDS of website the evidence came from
    fig 3  fragmentation     scan vs join per record + Def.-14 coverage maps
    fig 5  stopping rule     estimated undiscovered records shrinking under
                             the certification bar, per stratum
    fig 6  the answer        resolved records table + the CI decomposition

Usage (repo root; .env with OPENAI_API_KEY / SERPER_API_KEY / ANTHROPIC_API_KEY):

    python tests/demo_online_e2e.py                  # live run, low dollars
    python tests/demo_online_e2e.py --surface-er     # if torch isn't installed
    python tests/demo_online_e2e.py --selftest       # no network: exercises
                                                     # every figure/report path
                                                     # (full AND degraded data)

Budget shims follow the sanity_online_ch6 conventions: they truncate result
lists, cap fetches, sleep politely and downgrade one-source crashes to skips
(the Exp-1 lesson), but always DELEGATE to the real functions. Live output is
nondeterministic, so every check is an INVARIANT, never an exact value.

A run can come back THIN: funding-news domains (Crunchbase, TechCrunch,
Reuters, Bloomberg...) often refuse plain HTTP clients, so pages die at the
fetch stage before the pipeline ever sees them. The demo now narrates every
URL's fate live, splits the funnel at each leak point, and -- if zero facts
survive -- says so loudly with retry guidance instead of pretending success.
A degraded run FAILING its checks is the sanity check working.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # write PNGs; no display needed on any machine
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--query",
                default=("All funding rounds raised by Hugging Face: "
                         "the amount, date, and lead investor of each round."))
ap.add_argument("--attrs", default="amount,date,lead",
                help="comma-separated query attributes (the set A of Def. 14)")
ap.add_argument("--max-steps", type=int, default=3,
                help="frontier steps (= real searches issued)")
ap.add_argument("--k", type=int, default=6, help="results kept per search")
ap.add_argument("--max-fetches", type=int, default=18,
                help="global cap on real page fetch ATTEMPTS")
ap.add_argument("--sleep", type=float, default=0.8,
                help="seconds between fetches (politeness)")
ap.add_argument("--eps", type=float, default=0.10,
                help="completeness slack eps for the stopping rule / CI")
ap.add_argument("--surface-er", action="store_true",
                help="cluster by normalized surface form instead of the real "
                     "matcher (no torch needed; fragile pairs then vanish)")
ap.add_argument("--outdir", default="figures/demo_e2e",
                help="where the PNGs land")
ap.add_argument("--selftest", action="store_true",
                help="no network: run checks+figures on synthetic data")
ARGS = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
QUERY_ATTRS = {a.strip() for a in ARGS.attrs.split(",") if a.strip()}

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def info(msg: str) -> None:
    print("  info  " + msg)


# ===========================================================================
# 1. THE LIVE RUN  (budget-capped real end_to_end, ch6 shim conventions)
# ===========================================================================
def run_live():
    if not Path("prompts/relevance.txt").exists():
        sys.exit("Run from the repo root (prompts/ must be in the cwd).")
    missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY", "ANTHROPIC_API_KEY")
               if not os.environ.get(k)]
    if missing:
        from dotenv import load_dotenv
        load_dotenv()
        missing = [k for k in missing if not os.environ.get(k)]
    if missing:
        sys.exit(f"Live mode needs {missing} in the environment / .env")

    from webagg import pipeline as pipe
    from webagg import fetch as fetch_mod
    from webagg import extract as extract_mod
    from webagg.search import SerperBackend

    # searches issued / fetch attempts / pages that yielded text
    counters = {"searches": 0, "attempts": 0, "pages": 0}

    class BudgetedSearch:                       # real Serper, truncated
        def __init__(self):
            self._real = SerperBackend()

        def search(self, query, k=10, formulation_id=""):
            if counters["searches"] >= ARGS.max_steps:
                return []
            counters["searches"] += 1
            try:
                hits = self._real.search(query, k=ARGS.k,
                                         formulation_id=formulation_id)[:ARGS.k]
                info(f'search #{counters["searches"]}: {len(hits)} results '
                     f'for "{query[:60]}"')
                return hits
            except Exception as e:
                info(f"search failed, skipped: {e!r}")
                return []

    def budgeted_fetch(url, formulation_id):    # real HTTP, capped + polite,
        if counters["attempts"] >= ARGS.max_fetches:   # narrated per URL
            return None
        if url.lower().split("?")[0].endswith(
                (".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".zip",
                 ".doc", ".docx")):
            return None
        counters["attempts"] += 1
        time.sleep(ARGS.sleep)
        try:
            src = fetch_mod.fetch_url(url, formulation_id=formulation_id)
        except Exception as e:
            info(f"fetch skipped ({type(e).__name__}): {url[:60]}")
            return None
        if src is None:
            # the usual live-web attrition: bot walls, empty/JS-only pages
            info(f"no usable text (blocked/empty): {url[:70]}")
        else:
            counters["pages"] += 1
            info(f"fetched [{src.source_class}] {src.domain}")
        return src

    def hardened(fn, fallback, label):          # Exp-1: skip, never die
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:
                info(f"{label} failed on one source, skipped "
                     f"({type(e).__name__})")
                return fallback
        return inner

    cluster_fn = None
    if ARGS.surface_er:
        from webagg.frontier import normalize_surface

        def cluster_fn(mentions, sources):      # honest fallback: no torch,
            return {m.mention_id:               # but no fragile pairs either
                    f"{normalize_surface(m.entity_surface)}|{m.record_kind}"
                    for m in mentions}

    run_id = f"demo_e2e_{uuid.uuid4().hex[:8]}"
    empty_gate = {"n_a": 0, "n_b": 0, "agreed": 0, "disagreed": 0, "b_only": 0,
                  "validator_rejects": 0, "gate_abstains": 0}
    saved = {n: getattr(pipe, n) for n in
             ("SerperBackend", "fetch_url", "is_relevant", "extract_certified",
              "propose_followups")}
    real_followups = pipe.propose_followups
    try:
        pipe.SerperBackend = BudgetedSearch
        pipe.fetch_url = budgeted_fetch
        pipe.is_relevant = hardened(extract_mod.is_relevant, (False, 0.0),
                                    "relevance")
        pipe.extract_certified = hardened(extract_mod.extract_certified,
                                          ([], [], dict(empty_gate)),
                                          "extract_certified")
        pipe.propose_followups = hardened(
            lambda *a, **k: real_followups(*a, **k)[:2], [], "followups")

        print("=== live run " + "=" * 58)
        t0 = time.time()
        result = pipe.end_to_end(ARGS.query, run_id=run_id,
                                 query_attributes=QUERY_ATTRS,
                                 aggregate_attr="amount", eps=ARGS.eps,
                                 max_steps=ARGS.max_steps,
                                 cluster_fn=cluster_fn)
        elapsed = time.time() - t0
    finally:
        for n, v in saved.items():
            setattr(pipe, n, v)

    return result, run_id, counters, elapsed


# ===========================================================================
# 2. GATHER  (everything the checks and figures need, as plain data)
# ===========================================================================
def gather(result, run_id, counters, elapsed) -> dict:
    from webagg.storage import (get_session, load_sources, load_mentions,
                                MeasurementRow, RejectedSourceRow)
    # end_to_end persists everything under the run DB -- the DB is the source
    # of truth (impl §4.2), so reload from it rather than trusting in-memory
    db = Path(f"data/runs/{run_id}.sqlite")
    if not db.exists():
        raise SystemExit(f"run DB not found: {db}")
    session = get_session(str(db))

    # NOTE the persistence rule (pipeline §6.1): a Source ROW exists only for
    # pages that passed the relevance gate; rejected pages are kept whole in
    # rejected_sources for the phi-audit. So len(sources) = RELEVANT pages.
    sources = load_sources(session)
    n_rejected = session.query(RejectedSourceRow).count()
    mentions = [m for m in load_mentions(session) if m.accepted]

    rows = (session.query(MeasurementRow)
            .filter(MeasurementRow.run_id == run_id).all())

    def metric(name):
        return [r for r in rows if r.metric == name]

    uhat = defaultdict(list)                  # stratum -> [(step, U, psi, f1, N)]
    for r in metric("U_hat"):
        e = r.extra or {}
        uhat[r.stratum or "?"].append(
            (r.step, r.value, e.get("psi"), e.get("f1"), e.get("N")))
    for g in uhat:
        uhat[g].sort()

    stop = None
    if metric("stop"):
        r = metric("stop")[-1]
        stop = {"step": r.step, "reason": (r.extra or {}).get("reason")}

    frag_rows = [(r.extra or {}) for r in metric("frag_case")]
    prune = [(r.step, r.value, (r.extra or {}))
             for r in metric("single_class_prune")]
    weak = len(metric("weak_entity_link"))

    llm_tokens = defaultdict(lambda: [0, 0])  # purpose -> [in, out]
    for r in metric("llm_call"):
        e = r.extra or {}
        p = e.get("purpose", "?")
        llm_tokens[p][0] += int(e.get("input_tokens", 0) or 0)
        llm_tokens[p][1] += int(e.get("output_tokens", 0) or 0)

    records = []
    for rec in result["records"]:
        attrs = {}
        for a, cv in rec["attributes"].items():
            attrs[a] = {"value": cv.value, "belief": cv.belief,
                        "flags": list(cv.validator_flags)}
        records.append({"entity_id": rec["entity_id"],
                        "kind": rec["record_kind"],
                        "case": rec["frag_case"], "attrs": attrs})

    return {
        "query": ARGS.query, "elapsed": elapsed,
        "n_searches": counters["searches"],
        "n_attempts": counters["attempts"],   # fetches tried
        "n_pages": counters["pages"],         # pages that yielded text
        "n_rejected": n_rejected,             # ...but judged off-topic
        "n_relevant": len(sources),           # ...and kept as evidence
        "src_classes": [s.source_class or "other" for s in sources],
        "src_domains": [s.domain for s in sources],
        "n_mentions": len(mentions),
        "records": records, "frag_rows": frag_rows,
        "uhat": dict(uhat), "stop": stop, "prune": prune, "weak": weak,
        "ci": result["ci"], "answer": result["answer"],
        "llm_tokens": dict(llm_tokens), "eps": ARGS.eps,
    }


# ===========================================================================
# 3. CHECKS  (live invariants; structure and consistency, never exact values)
# ===========================================================================
def run_checks(d: dict) -> None:
    print("\n=== invariant checks " + "=" * 50)
    check(d["n_searches"] >= 1, f"real searches issued (n={d['n_searches']})")
    check(d["n_attempts"] >= 1, f"fetches attempted (n={d['n_attempts']})")
    check(d["n_pages"] >= 1,
          f"pages yielded text (n={d['n_pages']} of {d['n_attempts']} tried; "
          f"the difference is live-web attrition: bot walls, empty pages)")
    check(d["n_pages"] == d["n_relevant"] + d["n_rejected"],
          f"page accounting closed: {d['n_pages']} with text = "
          f"{d['n_relevant']} relevant + {d['n_rejected']} rejected "
          f"(rejections KEPT for the phi-audit)")
    check(all(c for c in d["src_classes"]),
          "every kept Source carries a source_class stamp (§12, fetch-time)")
    info(f"distinct source classes seen: {sorted(set(d['src_classes']))}")
    check(d["n_mentions"] >= 1,
          f"certified mentions extracted through the four-stage gate "
          f"(n={d['n_mentions']})")
    check(len(d["records"]) >= 1,
          f"records resolved by ER (n={len(d['records'])})")
    check(len(d["frag_rows"]) == len(d["records"]),
          "one Def.-14 fragmentation report logged per resolved record")
    ok_cases = {"scan_sufficient", "fragmented", "redundant", "empty"}
    check(all(fr.get("case") in ok_cases for fr in d["frag_rows"]),
          "every report lands in one of Def. 14's cases (or 'empty')")
    for fr in d["frag_rows"]:                  # U == union of K (Def. 14)
        union = sorted({a for attrs in fr.get("K", {}).values() for a in attrs})
        check(union == sorted(fr.get("U", [])),
              f"coverage-matrix consistency U = union(K) [{fr.get('record_id')}]")
    # weak_entity_link may only appear on fragmenting attributes
    for rec in d["records"]:
        rep = next((fr for fr in d["frag_rows"]
                    if fr.get("record_id") == f"{rec['entity_id']}/{rec['kind']}"),
                   {})
        for a, av in rec["attrs"].items():
            if "weak_entity_link" in av["flags"]:
                check(rep.get("case") == "fragmented",
                      f"guard flag only on fragmenting attrs ({a})")
    check(bool(d["uhat"]),
          f"per-stratum stopping series logged (strata={list(d['uhat'])})")
    for g, series in d["uhat"].items():
        check(all(u >= 0 and (p is None or p >= 0) for _, u, p, _, _ in series),
              f"U_hat and psi non-negative in stratum '{g}'")
    if d["stop"]:
        check(d["stop"]["reason"] in ("certified", "budget"),
              f"stop recorded honestly (reason={d['stop']['reason']})")
    else:
        info("no stop row: the step cap ended the run (expected in a "
             "budget-capped demo; a certificate needs more steps)")
    if d["prune"]:
        info(f"single-class prune fired: {d['prune']}")
    else:
        info("single-class prune did not fire -- by design: the floor is "
             "MIN_RECORDS_FOR_PRUNE=8 records and a short demo rarely gets "
             "there (documented conservative deviation)")
    ci = d["ci"]
    check(ci["ci_total"] >= 0 and ci["n_records"] == len(d["records"]),
          f"CI well-formed (answer={ci['answer']:.3g}, +/-{ci['ci_total']:.3g}, "
          f"n={ci['n_records']})")


# ===========================================================================
# 4. FIGURES  (plain-language; every one must survive an EMPTY run)
# ===========================================================================
CLS_ORDER = ["regulatory", "vendor", "news", "investor", "social", "other"]
CLS_COLOR = {"regulatory": "#4c72b0", "vendor": "#dd8452", "news": "#55a868",
             "investor": "#c44e52", "social": "#8172b3", "other": "#937860"}


def _save(fig, outdir: Path, name: str) -> None:
    fig.tight_layout()
    path = outdir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  fig   {path}")


def _placeholder(outdir, name, text):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, wrap=True)
    _save(fig, outdir, name)


def fig_funnel(d, outdir):
    stages = ["searches\nissued", "fetches\nattempted", "pages\nwith text",
              "judged\nrelevant", "facts\ncertified", "records\nresolved"]
    vals = [d["n_searches"], d["n_attempts"], d["n_pages"],
            d["n_relevant"], d["n_mentions"], len(d["records"])]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    bars = ax.bar(stages, vals,
                  color=["#999", "#bbb", "#4c72b0", "#8172b3", "#55a868",
                         "#dd8452"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f" {v}",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("The pipeline funnel: raw web -> certified, resolved records\n"
                 "(each drop is a DELIBERATE filter or honest live-web attrition)")
    ax.set_ylabel("count")
    _save(fig, outdir, "fig1_funnel.png")


def fig_source_classes(d, outdir):
    counts = defaultdict(int)
    for c in d["src_classes"]:
        counts[c] += 1
    if not counts:
        _placeholder(outdir, "fig2_source_classes.png",
                     "No pages survived to be classified in this run --\n"
                     "see the funnel for where the evidence thinned out.")
        return
    cls = [c for c in CLS_ORDER if counts.get(c)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cls, [counts[c] for c in cls], color=[CLS_COLOR[c] for c in cls])
    ax.set_title("Where the evidence came from\n"
                 "(each page auto-classified at fetch time -- ch. 12, no LLM)")
    ax.set_ylabel("relevant pages kept")
    _save(fig, outdir, "fig2_source_classes.png")


def fig_fragmentation(d, outdir):
    case_counts = defaultdict(int)
    for fr in d["frag_rows"]:
        case_counts[fr.get("case", "?")] += 1
    labels = {"scan_sufficient": "SCAN: one source class\nhad every attribute",
              "fragmented": "JOIN: attributes split\nacross classes",
              "redundant": "REDUNDANT: every attribute\nin 2+ classes",
              "empty": "empty"}
    keys = [k for k in ("scan_sufficient", "fragmented", "redundant", "empty")
            if case_counts.get(k)]
    if not keys:
        _placeholder(outdir, "fig3_fragmentation_cases.png",
                     "No resolved records, so no scan-vs-join decisions "
                     "to show in this run.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([labels[k] for k in keys], [case_counts[k] for k in keys],
           color=["#55a868", "#c44e52", "#4c72b0", "#999"][:len(keys)])
    ax.set_title("Join or scan? -- the ch. 12 routing decision per record")
    ax.set_ylabel("records")
    _save(fig, outdir, "fig3_fragmentation_cases.png")

    # Def.-14 coverage maps for the richest records
    rich = sorted(d["frag_rows"], key=lambda fr: -len(fr.get("U", [])))[:3]
    rich = [fr for fr in rich if fr.get("U")]
    if not rich:
        return
    fig, axes = plt.subplots(1, len(rich), figsize=(4.6 * len(rich), 3.8),
                             squeeze=False)
    for ax, fr in zip(axes[0], rich):
        attrs = sorted(fr["U"])
        K = fr.get("K", {})
        cls = [c for c in CLS_ORDER if c in K]
        grid = [[1 if a in K.get(c, []) else 0 for c in cls] for a in attrs]
        ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cls)), cls, rotation=30, ha="right")
        ax.set_yticks(range(len(attrs)), attrs)
        for i in range(len(attrs)):
            for j in range(len(cls)):
                ax.text(j, i, "x" if grid[i][j] else "", ha="center",
                        va="center", fontweight="bold")
        title = fr.get("record_id", "?")
        verdict = fr.get("case")
        if verdict == "scan_sufficient":
            verdict += f" ({fr.get('scan_class')})"
        ax.set_title(f"{title}\n-> {verdict}", fontsize=9)
    fig.suptitle("Coverage matrix M(rho): which source class supplied "
                 "which attribute (paper Def. 14)")
    _save(fig, outdir, "fig4_coverage_matrix.png")


def fig_stopping(d, outdir):
    if not d["uhat"]:
        _placeholder(outdir, "fig5_stopping_rule.png",
                     "No discovery strata formed in this run (no certified "
                     "facts),\nso there is no stopping series to plot.")
        return
    strata = list(d["uhat"])[:4]
    fig, axes = plt.subplots(1, len(strata), figsize=(4.6 * len(strata), 3.8),
                             squeeze=False, sharex=True)
    for ax, g in zip(axes[0], strata):
        s = d["uhat"][g]
        steps = [t for t, *_ in s]
        ax.plot(steps, [u for _, u, *_ in s], "o-", color="#c44e52",
                label="U_hat: est. records\nstill undiscovered")
        psis = [(t, p) for t, _, p, _, _ in s if p is not None]
        if psis:
            ax.plot([t for t, _ in psis], [p for _, p in psis], "s--",
                    color="#4c72b0", label="psi: confidence radius")
        if d["stop"] and d["stop"]["reason"] == "certified":
            ax.axvline(d["stop"]["step"], color="#55a868", ls=":",
                       label=f"stopped: {d['stop']['reason']}")
        ax.set_title(f"stratum: {g}", fontsize=9)
        ax.set_xlabel("agent step")
        ax.legend(fontsize=7)
    fig.suptitle("The stopping rule: search ends only when the estimated "
                 "undiscovered mass is provably small (per stratum)")
    _save(fig, outdir, "fig5_stopping_rule.png")


def fig_answer(d, outdir):
    if not d["records"]:
        _placeholder(outdir, "fig6_answer_and_ci.png",
                     "No records were resolved in this run, so there is no "
                     "answer to certify.\nSee the funnel for where the "
                     "evidence thinned out, then re-run\n(search results are "
                     "nondeterministic; consider --k 8 --max-fetches 24).")
        return
    fig = plt.figure(figsize=(9.5, 1.6 + 0.5 * max(len(d["records"]), 4)))
    ax = fig.add_subplot(211)
    ax.axis("off")
    attrs = sorted({a for r in d["records"] for a in r["attrs"]})
    header = ["record", "route"] + attrs
    cells = []
    for r in d["records"][:10]:
        row = [f"{r['entity_id']}/{r['kind']}"[:34],
               {"scan_sufficient": "scan", "fragmented": "join",
                "redundant": "redund.", "empty": "-"}.get(r["case"], "?")]
        for a in attrs:
            av = r["attrs"].get(a)
            if not av:
                row.append("-")
            else:
                flag = " !" if "weak_entity_link" in av["flags"] else ""
                row.append(f"{str(av['value'])[:16]} (b={av['belief']:.2f}{flag})")
        cells.append(row)
    tbl = ax.table(cellText=cells, colLabels=header, loc="center",
                   cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    ax.set_title("Resolved records: adopted value per attribute, with its "
                 "corroborated belief b\n(! = weak entity link: the ch. 12 "
                 "guard halved belief and flagged it for verification)",
                 fontsize=9)

    ax2 = fig.add_subplot(212)
    ci = d["ci"]
    parts = ["value\nuncertainty", "entity-resolution\nrisk",
             "possible missed\nrecords"]
    vals = [ci["value_term"], ci["join_term"], ci["recall_term"]]
    ax2.barh(parts, vals, color=["#4c72b0", "#dd8452", "#c44e52"])
    ax2.set_title(f"The certified answer: SUM(amount) = {ci['answer']:,.0f}  "
                  f"+/- {ci['ci_total']:,.0f}\n"
                  f"-- and where the +/- comes from (every risk priced in)",
                  fontsize=9)
    ax2.set_xlabel("contribution to the confidence interval")
    _save(fig, outdir, "fig6_answer_and_ci.png")


def make_figures(d: dict, outdir: Path | None = None) -> None:
    outdir = outdir or Path(ARGS.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("\n=== figures " + "=" * 59)
    fig_funnel(d, outdir)
    fig_source_classes(d, outdir)
    fig_fragmentation(d, outdir)
    fig_stopping(d, outdir)
    fig_answer(d, outdir)


# ===========================================================================
# 5. PLAIN-LANGUAGE SUMMARY  (the demo narrative, printed last)
# ===========================================================================
def summary(d: dict) -> None:
    print("\n=== what this run demonstrated " + "=" * 40)
    print(f'  query: "{d["query"]}"')
    print(f"  1. DISCOVER   {d['n_searches']} live searches; "
          f"{d['n_attempts']} fetches tried -> {d['n_pages']} pages with "
          f"text -> {d['n_relevant']} judged relevant "
          f"({d['n_rejected']} rejections kept for the phi-audit)")
    print(f"  2. CERTIFY    {d['n_mentions']} facts survived the four-stage "
          f"extraction gate (dual extraction, validators, conformal gate)")
    print(f"  3. TRACK      per-stratum discovery curves logged every step; "
          f"the agent stops itself only when undiscovered mass is provably "
          f"small" + (f" (stopped: {d['stop']['reason']})" if d["stop"]
                      else " (step cap reached first in this small demo)"))
    print(f"  4. ROUTE      {len(d['records'])} resolved records, each "
          f"routed scan-vs-join from its Def.-14 coverage matrix "
          f"(ch. 12); weak-entity-link guard fired {d['weak']} time(s)")
    print(f"  5. ANSWER     SUM = {d['ci']['answer']:,.0f} "
          f"+/- {d['ci']['ci_total']:,.0f}, the +/- decomposed into value / "
          f"ER / recall risk -- an answer with an honest error bar")
    if d["llm_tokens"]:
        tot_in = sum(v[0] for v in d["llm_tokens"].values())
        tot_out = sum(v[1] for v in d["llm_tokens"].values())
        print(f"  cost: {tot_in} in / {tot_out} out tokens across "
              f"{len(d['llm_tokens'])} LLM purposes; {d['elapsed']:.0f}s wall")

    if d["n_mentions"] == 0:
        print("\n  *** DEGRADED RUN -- the checks refused it, as designed ***")
        if d["n_pages"] == 0:
            print("  every fetch died at the wall: the top results were "
                  "bot-blocking domains.")
        elif d["n_relevant"] == 0:
            print("  pages fetched but none judged on-topic: likely "
                  "aggregator/SEO pages in the top results.")
        else:
            print("  relevant pages yielded no certified facts: the strict "
                  "dual-extraction gate abstained.")
        print("  Live search results are nondeterministic. Re-run as-is, or "
              "widen the net:\n"
              "      python tests/demo_online_e2e.py --k 8 --max-fetches 24 "
              "--max-steps 4\n"
              "  A clean pass on retry is normal; repeated zero-fact runs "
              "would be real signal.")
    print("\n" + ("SOME CHECKS FAILED:\n  - " + "\n  - ".join(FAILURES)
                  if FAILURES else "ALL CHECKS PASSED"))


# ===========================================================================
# 6. SELFTEST  (no network: figure/report paths on FULL and DEGRADED data)
# ===========================================================================
def synthetic() -> dict:
    uhat = {"2024|series_b": [(1, 6.0, 4.0, 5, 8), (2, 2.5, 1.8, 2, 12),
                              (3, 0.7, 0.6, 1, 14)],
            "2025|series_c": [(2, 3.0, 2.5, 3, 5), (3, 0.9, 0.8, 1, 7)]}
    frag_rows = [
        {"record_id": "acme/round_b", "case": "fragmented",
         "U": ["amount", "date", "lead", "employees"],
         "K": {"vendor": ["amount", "date", "lead"], "news": ["amount", "date"],
               "social": ["employees"]}},
        {"record_id": "beta/round_a", "case": "scan_sufficient",
         "scan_class": "vendor",
         "U": ["amount", "date", "lead"],
         "K": {"vendor": ["amount", "date", "lead"], "news": ["amount"]}},
        {"record_id": "gama/round_a", "case": "redundant",
         "U": ["amount", "date"],
         "K": {"vendor": ["amount", "date"], "news": ["amount", "date"]}},
    ]
    records = [
        {"entity_id": "acme", "kind": "round_b", "case": "fragmented",
         "attrs": {"amount": {"value": "$40M", "belief": 0.91, "flags": []},
                   "date": {"value": "2025-03-02", "belief": 0.88, "flags": []},
                   "lead": {"value": "Sequoia", "belief": 0.83, "flags": []},
                   "employees": {"value": "120", "belief": 0.35,
                                 "flags": ["weak_entity_link"]}}},
        {"entity_id": "beta", "kind": "round_a", "case": "scan_sufficient",
         "attrs": {"amount": {"value": "$12M", "belief": 0.86, "flags": []},
                   "date": {"value": "2024-11-10", "belief": 0.84, "flags": []},
                   "lead": {"value": "Accel", "belief": 0.80, "flags": []}}},
        {"entity_id": "gama", "kind": "round_a", "case": "redundant",
         "attrs": {"amount": {"value": "$8M", "belief": 0.93, "flags": []},
                   "date": {"value": "2024-06-01", "belief": 0.90, "flags": []}}},
    ]
    return {"query": "(selftest) synthetic funding rounds", "elapsed": 0.0,
            "n_searches": 3, "n_attempts": 16, "n_pages": 11,
            "n_rejected": 3, "n_relevant": 8,
            "src_classes": ["vendor"] * 3 + ["news"] * 3 + ["regulatory"]
                            + ["social"],
            "src_domains": [], "n_mentions": 23,
            "records": records, "frag_rows": frag_rows, "uhat": uhat,
            "stop": {"step": 3, "reason": "certified"},
            "prune": [], "weak": 1,
            "ci": {"answer": 60_000_000, "n_records": 3,
                   "value_term": 1_800_000, "join_term": 900_000,
                   "recall_term": 6_000_000, "ci_total": 8_700_000},
            "answer": 60_000_000,
            "llm_tokens": {"extract": [9000, 2000], "relevance": [1200, 60]},
            "eps": 0.10}


def synthetic_degraded() -> dict:
    """The thin-run shape the 2026-08 live run produced: 1 page, 0 facts.
    Every figure must render a placeholder instead of crashing."""
    d = synthetic()
    d.update({"query": "(selftest) degraded thin run", "n_attempts": 15,
              "n_pages": 1, "n_rejected": 0, "n_relevant": 1,
              "src_classes": ["other"], "n_mentions": 0, "records": [],
              "frag_rows": [], "uhat": {}, "stop": None, "weak": 0,
              "ci": {"answer": 0, "n_records": 0, "value_term": 0,
                     "join_term": 0, "recall_term": 0, "ci_total": 0},
              "answer": 0})
    return d


# ===========================================================================
def main() -> None:
    if ARGS.selftest:
        print("SELFTEST 1/2: full synthetic data")
        d = synthetic()
        run_checks(d)
        make_figures(d, Path(ARGS.outdir))
        summary(d)
        print("\nSELFTEST 2/2: DEGRADED synthetic data (must not crash; "
              "its check failures are expected and not counted)")
        n_fail = len(FAILURES)
        dd = synthetic_degraded()
        run_checks(dd)
        make_figures(dd, Path(ARGS.outdir) / "degraded")
        summary(dd)
        del FAILURES[n_fail:]        # degraded-mode failures are the POINT
        sys.exit(1 if FAILURES else 0)

    result, run_id, counters, elapsed = run_live()
    print(f"\n  run_id: {run_id}   (DB + measurements kept under data/runs/)")
    d = gather(result, run_id, counters, elapsed)
    run_checks(d)
    make_figures(d)
    summary(d)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
