"""ONLINE sanity check for §11 (claims engine: checksums + gap direction),
real APIs. The live counterpart of tests/test_claims_ch11.py: the offline
suite proves the LOGIC (fixed claims -> known verdicts); this proves the
SEAMS (real pages yield well-formed Claims, the real LLM turns a gap
payload into usable formulations, a real run leaves the §11 audit trail
in the DB).

Run from the repo root (.env with OPENAI_API_KEY, SERPER_API_KEY,
ANTHROPIC_API_KEY must be present):

    python tests/sanity_online_ch11.py               # tier 1 only:
                                                     # 1 gap-formulation LLM
                                                     # call, 1 search, up to
                                                     # --pages fetches + 2
                                                     # extraction calls each.
                                                     # Cents.
    python tests/sanity_online_ch11.py --mini-run    # + tier 2: a real
                                                     # budget-capped
                                                     # run_query(). ~30-60
                                                     # LLM calls. Low dollars.

Tier 1 -- the three §11 seams, each touched once:
    gap-direction  a REAL LLM call through prompts/gap_formulations.txt on
                   the worked example's residual (1 round / ~$11M missing,
                   2021+2025 covered): Formulations come back well-formed,
                   the insert-once dedup holds, the call lands in
                   measurements. Falling back to the ch. 7 template is a
                   FAIL here -- the fallback is what the offline tests
                   already cover; this tier exists to prove the live path.
    live claims    real Serper search aimed at aggregate statements
                   ("<entity> total funding raised"), real fetches, real
                   extract_certified: every Claim that comes back is
                   well-formed and ingests cleanly. Zero claims is a WARN,
                   not a FAIL (aggregate statements are much rarer than
                   record mentions; try --pages 5).
    checksum       the engine run on whatever live claims arrived:
                   corroborated() shapes, a matching view certifies
                   (rule R1: belief stored), a deficit view yields the
                   count_gap payload + arms the App. E brake iff belief
                   clears CLAIM_BRAKE_MIN_BELIEF, a SUM residual yields
                   the Definition-12 payload. Plus the rule-R2 demotion
                   report. No LLM calls in this part.

Tier 2 -- a real (tiny) run_query() with the same budget shims as
tests/sanity_online_ch6.py, but a query phrased to elicit aggregate
claims. Afterwards, the §11 audit trail is read back from the DB:
claims persisted, claim_demotion_rate logged every step, certified
strata carry their conditional terms (cert_kind/belief/delta_plus --
rule R1), conflicts carry their payloads (rule R3), gap-directed
formulations are marked in the formulations table (Definition 12).

Live output is nondeterministic, so every check is an INVARIANT
(structure, ranges, consistency), never an exact value.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--mini-run", action="store_true",
                help="also run tier 2: a budget-capped real run_query()")
ap.add_argument("--entity", default="Hugging Face",
                help="entity used for the live-claims search and tier 2")
ap.add_argument("--query", default=None,
                help="tier 2 query (default: an aggregate-eliciting question "
                     "about --entity)")
ap.add_argument("--pages", type=int, default=3,
                help="tier 1: pages fetched+extracted for live claims")
ap.add_argument("--max-steps", type=int, default=2,
                help="tier 2: frontier steps (= searches issued)")
ap.add_argument("--k", type=int, default=4,
                help="tier 2: results kept per search (bounds fetch volume)")
ap.add_argument("--max-fetches", type=int, default=10,
                help="tier 2: global cap on page fetches")
ap.add_argument("--sleep", type=float, default=0.8,
                help="politeness delay between fetches (seconds)")
ARGS = ap.parse_args()
if ARGS.query is None:
    ARGS.query = (f"How many funding rounds has {ARGS.entity} raised in "
                  f"total, and for what total amount? List every round "
                  f"with its amount, date, and lead investor.")

if not Path("prompts/gap_formulations.txt").exists():
    sys.exit("Run from the repo root (prompts/ must be in the cwd).")
missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY", "ANTHROPIC_API_KEY")
           if not os.environ.get(k)]
if missing:
    from dotenv import load_dotenv
    load_dotenv()
    missing = [k for k in missing if not os.environ.get(k)]
if missing:
    sys.exit(f"Live mode needs {missing} in the environment / .env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webagg import config                                   # noqa: E402
from webagg import pipeline as pipe                         # noqa: E402
from webagg import fetch as fetch_mod                       # noqa: E402
from webagg import extract as extract_mod                   # noqa: E402
from webagg.llm import set_llm_logger                       # noqa: E402
from webagg.search import SerperBackend                     # noqa: E402
from webagg.calibration import ConformalGate                # noqa: E402
from webagg.claims import ClaimsEngine, CoverageView        # noqa: E402
from webagg.frontier import (FrontierState, StratumState,   # noqa: E402
                             normalize_surface)
from webagg.storage import (get_session, load_claims,       # noqa: E402
                            MeasurementRow, StratumStateRow,
                            FrontierFormulationRow)

FAILURES: list[str] = []
# the ch. 7 fallback template, verbatim (claims.py gap_formulations except-
# branch). Seeing it in tier 1 means the LIVE path did not run.
FALLBACK_TEMPLATE = '"{g}" additional funding round announcement'


def check(cond: bool, msg: str):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def info(msg: str):
    print("  info  " + msg)


# ===========================================================================
# TIER 1: the three §11 seams
# ===========================================================================
def tier1() -> None:
    Path("data/runs").mkdir(parents=True, exist_ok=True)
    tmp_db = f"data/runs/sanity11_{uuid.uuid4().hex[:8]}.sqlite"
    session = get_session(tmp_db)
    ce = ClaimsEngine(session)
    g = normalize_surface(ARGS.entity)

    # ---- 1a. gap-directed formulations: the REAL LLM on the worked gap ----
    print("\n=== tier 1a: gap formulations (real LLM) " + "=" * 29)
    set_llm_logger(session, "sanity11_tier1")
    # the guide's worked residual, with EXACTLY the keys checksum() emits
    # (Definition 12): one round, ~$11M, eras 2021+2025 and early stages
    # covered -- so the missing round is pre-2021, likely seed, ~$11M.
    gap = {"count_gap": 1, "value_gap": 11e6,
           "eras_covered": [2021, 2025], "stages": ["series a", "series b"]}
    fs = ce.gap_formulations(g, gap)
    set_llm_logger(None, "")
    session.commit()

    check(1 <= len(fs) <= 4, f"1-4 formulations returned (n={len(fs)})")
    for f in fs:
        check(isinstance(f.query, str) and len(f.query.strip()) > 0,
              f"non-empty query: {f.query[:70]!r}")
        check(f.stratum == g and f.gap_directed,
              "tagged with the stratum + gap_directed=True (Def. 12 marker)")
        check(0.0 < f.p_success <= 1.0 and f.yield_if_success > 0,
              f"sane frontier estimates (p={f.p_success}, "
              f"yield={f.yield_if_success})")
    check(not (len(fs) == 1 and fs[0].query == FALLBACK_TEMPLATE.format(g=g)),
          "LIVE LLM path ran (not the ch. 7 offline fallback template -- "
          "if this fails, check OPENAI_API_KEY / connectivity)")
    # soft content check: did the LLM actually USE the constraints?
    used = any(("11" in f.query) or ("seed" in f.query.lower())
               or any(str(y) in f.query for y in range(2015, 2021))
               for f in fs)
    if used:
        check(True, "at least one query targets the residual "
                    "(value/era/stage constraint visibly used)")
    else:
        info("no query visibly used the value/era/stage constraints -- "
             "legal (prompt says 'where known') but worth eyeballing:")
        for f in fs:
            info(f"    {f.query}")
    check(ce.gap_formulations(g, gap) == [],
          "insert-once dedup: same (stratum, gap signature) emits nothing "
          "(guide §7.3: re-detected gaps must not balloon the frontier)")
    row = (session.query(MeasurementRow).filter_by(metric="llm_call")
           .order_by(MeasurementRow.id.desc()).first())
    check(row is not None and row.extra["purpose"] == "gap_formulations",
          "the LLM call landed in measurements (purpose=gap_formulations)")

    # ---- 1b. live claims: real search -> fetch -> extract -> ingest -------
    print("\n=== tier 1b: live claims from real pages " + "=" * 29)
    results = SerperBackend().search(f"{ARGS.entity} total funding raised",
                                     k=max(6, ARGS.pages),
                                     formulation_id="f_sanity11")
    check(len(results) >= 1, f"search returned results (n={len(results)})")
    gate = ConformalGate(delta_E=config.DELTA_E)   # unfitted: bootstrap mode
    n_pages, n_claims = 0, 0
    for r in results:
        if n_pages >= ARGS.pages:
            break
        url = r["url"]
        if url.lower().split("?")[0].endswith(
                (".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".zip",
                 ".doc", ".docx")):
            continue
        try:
            src = fetch_mod.fetch_url(url, formulation_id="f_sanity11")
        except Exception as e:
            info(f"fetch skipped ({type(e).__name__}): {url[:60]}")
            continue
        if src is None:
            continue
        n_pages += 1
        ce.register_source(src)          # enables FULL §8 corroboration
        time.sleep(ARGS.sleep)
        try:
            _, claims, _ = extract_mod.extract_certified(src, ARGS.query,
                                                         gate=gate)
        except Exception as e:
            info(f"extract skipped ({type(e).__name__}): {url[:60]}")
            continue
        for c in claims:
            check(c.functional in ("SUM", "COUNT"),
                  f"claim functional in {{SUM,COUNT}} ({c.functional})")
            check(c.value_num > 0,
                  f"claim value positive ({c.functional}={c.value_num:g})")
            check(bool(c.stratum_surface.strip()),
                  f"claim scoped to a surface ({c.stratum_surface!r})")
            check(c.source_id == src.source_id,
                  "claim provenance points at its fetched Source")
            ce.ingest(c)
            n_claims += 1
            info(f"claim: {c.stratum_surface!r} {c.functional} "
                 f"{c.value_num:g} scope={c.scope!r}")
        info(f"page {n_pages}: {len(claims)} claim(s) from {url[:60]}")
    session.commit()
    if n_claims == 0:
        info("WARN: 0 claims from these pages. Aggregate statements are "
             "much rarer than record mentions; not a failure. Re-run with "
             "--pages 5, or rely on tier 2 for volume.")
        return

    # ---- 1c. corroborate + checksum on the live claims (no LLM) -----------
    print("\n=== tier 1c: corroboration + checksum on live claims " + "=" * 17)
    # richest stratum = the one most claims landed under
    g = max(ce._claims, key=lambda k: len(ce._claims[k]))
    info(f"stratum under test: {g!r} ({len(ce._claims[g])} claim(s))")
    cc = ce.corroborated(g, "COUNT")
    sc = ce.corroborated(g, "SUM")
    for name, out in (("COUNT", cc), ("SUM", sc)):
        if out is None:
            info(f"{name}: no clean corroborated claim (fine live)")
            continue
        v, b, tol = out
        check(v > 0 and 0.0 < b <= 1.0 and tol >= 0.0,
              f"{name} corroborated: value={v:g}, belief={b:.3f}, "
              f"tol={tol:g} (shapes sane)")
    if cc:
        n, b_n, _ = int(round(cc[0])), cc[1], cc[2]
        # matching view -> Thm 4(b) certification, belief STORED (rule R1)
        st = ce.checksum(g, CoverageView(n_records=n))
        check(st.certified and st.kind == "COUNT" and st.belief == b_n,
              f"matching view certifies COUNT with belief stored "
              f"(b={st.belief:.3f})")
        # deficit view -> hard-brake gap payload (Thm 4b / App. E)
        st = ce.checksum(g, CoverageView(n_records=max(0, n - 1)))
        check((not st.certified) and st.gap is not None
              and st.gap.get("count_gap") == 1,
              "deficit view yields gap={'count_gap': 1} (the brake + steer)")
        # brake arms iff belief clears the threshold (§11 config)
        state = FrontierState()
        state.strata[g] = StratumState(name=g)
        ce.update_cardinality_brakes(state)
        armed = state.strata[g].claimed_count is not None
        check(armed == (b_n >= config.CLAIM_BRAKE_MIN_BELIEF),
              f"App. E brake armed iff belief >= "
              f"{config.CLAIM_BRAKE_MIN_BELIEF} (b={b_n:.3f}, armed={armed})")
    if sc and not cc:
        V, b_V, tol = sc
        # a view assembled to exactly V -> Thm 4(a) certification
        st = ce.checksum(g, CoverageView(n_records=0, sum=V))
        check(st.certified and st.kind == "SUM" and st.delta_plus is not None
              and st.delta_plus <= ce.tol_rel * V,
              f"matching SUM view certifies (Delta+={st.delta_plus:g} <= "
              f"{ce.tol_rel:.0%} of V)")
    if sc:
        V = sc[0]
        # half-assembled view -> Definition-12 residual payload
        st2 = ce.checksum(g, CoverageView(n_records=(int(round(cc[0])) if cc
                                                     else 0),
                                          sum=V / 2,
                                          years=(2021,), stages=("series a",)))
        if not st2.certified:
            check(st2.gap is not None and "value_gap" in st2.gap
                  and "eras_covered" in st2.gap and "stages" in st2.gap,
                  "SUM residual carries the full Definition-12 payload "
                  "(value_gap + covered eras/stages)")

    # ---- 1d. rule-R2 discipline report ------------------------------------
    dr = ce.demotion_rate
    check(0.0 <= dr <= 1.0, f"demotion rate in [0,1] ({dr:.2f})")
    for d in ce.demotions:
        info(f"demoted (rule R2): {d}")
    for c in ce.conflicts:
        info(f"conflict (rule R3): {c}")
    info(f"tier 1 DB: {tmp_db}")


# ===========================================================================
# TIER 2: budget-capped real run_query(), then the §11 audit trail
# ===========================================================================
class BudgetedSearch:
    """Real Serper, truncated to k and capped at max_steps searches."""
    def __init__(self):
        self._real = SerperBackend()
        self.searches = 0

    def search(self, query, k=10, formulation_id=""):
        if self.searches >= ARGS.max_steps:
            return []
        self.searches += 1
        try:
            return self._real.search(query, k=ARGS.k,
                                     formulation_id=formulation_id)[:ARGS.k]
        except Exception as e:
            print(f"  info  search failed, skipped: {e!r}")
            return []


def make_budgeted_fetch(real_fetch):
    state = {"fetches": 0}

    def budgeted(url, formulation_id):
        if state["fetches"] >= ARGS.max_fetches:
            return None
        if url.lower().split("?")[0].endswith(
                (".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".zip",
                 ".doc", ".docx")):
            return None
        state["fetches"] += 1
        time.sleep(ARGS.sleep)                       # politeness
        try:
            return real_fetch(url, formulation_id=formulation_id)
        except Exception as e:
            print(f"  info  fetch skipped ({type(e).__name__}): {url[:60]}")
            return None
    return budgeted


def make_hardened(fn, fallback, label):
    """One bad source downgrades to a skip, never a dead run (Exp-1 lesson).
    Delegates to the REAL function."""
    def hardened(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            print(f"  info  {label} failed on one source, skipped "
                  f"({type(e).__name__})")
            return fallback
    return hardened


def tier2() -> None:
    print("\n=== tier 2: budget-capped REAL run_query() " + "=" * 27)
    run_id = f"sanity11_live_{uuid.uuid4().hex[:8]}"
    saved = {n: getattr(pipe, n) for n in
             ("SerperBackend", "fetch_url", "is_relevant", "extract_certified",
              "propose_followups")}
    try:
        pipe.SerperBackend = BudgetedSearch
        pipe.fetch_url = make_budgeted_fetch(fetch_mod.fetch_url)
        pipe.is_relevant = make_hardened(extract_mod.is_relevant,
                                         (False, 0.0), "relevance")
        pipe.extract_certified = make_hardened(extract_mod.extract_certified,
                                               ([], [], {"n_a": 0, "n_b": 0,
                                                         "agreed": 0,
                                                         "disagreed": 0,
                                                         "b_only": 0,
                                                         "validator_rejects": 0,
                                                         "gate_abstains": 0}),
                                               "extract_certified")
        real_followups = pipe.propose_followups
        pipe.propose_followups = make_hardened(
            lambda *a, **k: real_followups(*a, **k)[:2], [], "followups")

        t0 = time.time()
        state, session = pipe.run_query(ARGS.query, run_id=run_id,
                                        max_steps=ARGS.max_steps)
        elapsed = time.time() - t0
    finally:
        for n, v in saved.items():
            setattr(pipe, n, v)

    # ---- the §11 audit trail, read back from the DB -----------------------
    print("\n=== tier 2: §11 audit trail (from the DB) " + "=" * 28)
    rows = session.query(MeasurementRow).filter_by(run_id=run_id).all()
    claims = load_claims(session)
    info(f"{len(claims)} claim(s) persisted, wall {elapsed:.0f}s, "
         f"DB data/runs/{run_id}.sqlite")
    for c in claims:
        check(c.functional in ("SUM", "COUNT") and c.value_num > 0
              and bool(c.stratum_surface.strip()),
              f"persisted claim well-formed ({c.stratum_surface!r} "
              f"{c.functional} {c.value_num:g})")

    # rule R2: the demotion rate is logged EVERY step, always in [0,1]
    dr_rows = [r for r in rows if r.metric == "claim_demotion_rate"]
    check(len(dr_rows) >= 1,
          f"claim_demotion_rate logged every step (n={len(dr_rows)})")
    check(all(0.0 <= r.value <= 1.0 for r in dr_rows),
          "all demotion rates in [0,1]")

    # rule R1: any checksum-certified stratum snapshot carries its
    # conditional terms; kind SUM additionally carries Delta+
    srows = session.query(StratumStateRow).filter_by(run_id=run_id).all()
    cert = [r for r in srows if r.certified == "checksum"]
    for r in cert:
        check(r.cert_kind in ("COUNT", "SUM")
              and r.cert_belief is not None and 0.0 < r.cert_belief <= 1.0,
              f"certified stratum {r.stratum!r} stores kind+belief "
              f"({r.cert_kind}, b={r.cert_belief})")
        if r.cert_kind == "SUM":
            check(r.cert_delta_plus is not None and r.cert_delta_plus >= 0,
                  f"SUM cert stores Delta+ ({r.cert_delta_plus})")
    cert_ms = [r for r in rows if r.metric in ("checksum_certified",
                                               "checksum_certified_post_er")]
    if cert:
        check(len(cert_ms) >= 1,
              "certified snapshot(s) have a matching checksum_certified "
              "measurement")
    if not cert and not cert_ms:
        info("no checksum certification this run (normal at this budget: "
             "certification needs a corroborated claim AND a matching "
             "assembled view)")

    # rule R3: any conflict measurement carries its payload
    for r in (r for r in rows if r.metric == "claim_conflict"):
        check(isinstance(r.extra.get("conflicts"), list)
              and len(r.extra["conflicts"]) >= 1,
              f"conflict on {r.extra.get('stratum') or r.stratum!r} carries "
              f"its verification payload")

    # Definition 12: gap-directed formulations are marked in the DB
    gap_f = (session.query(FrontierFormulationRow)
             .filter_by(run_id=run_id, gap_directed=1).all())
    for f in gap_f[:5]:
        check(bool(f.query.strip()) and f.stratum,
              f"gap-directed formulation well-formed: {f.query[:60]!r}")
    if not gap_f:
        info("no gap-directed formulations this run (normal: needs a "
             "corroborated claim whose checksum does NOT close)")

    # ---- §11 activity summary ---------------------------------------------
    print("\n=== §11 activity summary " + "=" * 45)
    info(f"claims ingested: {len(claims)}  |  "
         f"final demotion rate: {dr_rows[-1].value if dr_rows else 'n/a'}")
    info(f"checksum certifications: {len(cert_ms)}  |  "
         f"conflicts: {sum(1 for r in rows if r.metric == 'claim_conflict')}"
         f"  |  gap-directed formulations: {len(gap_f)}")
    for r in cert_ms:
        info(f"  certified: stratum={r.stratum!r} kind="
             f"{r.extra.get('kind')} b={r.value:.3f} "
             f"delta_plus={r.extra.get('delta_plus')}")
    for f in gap_f[:5]:
        info(f"  gap query: {f.query[:70]!r} (p={f.p_success}, "
             f"yield={f.yield_if_success})")


# ===========================================================================
if __name__ == "__main__":
    tier1()
    if ARGS.mini_run:
        tier2()
    print(f"\n{len(FAILURES)} CHECK(S) FAILED" if FAILURES else
          "\nALL CHECKS PASSED")
    sys.exit(1 if FAILURES else 0)
