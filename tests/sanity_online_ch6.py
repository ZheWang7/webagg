"""ONLINE sanity check for chapters 3-6 (post-SIGMOD wiring), real APIs.

Two tiers, both run from the repo root (.env with OPENAI_API_KEY,
SERPER_API_KEY, ANTHROPIC_API_KEY must be present):

    python tests/sanity_online_ch6.py                # tier 1 only: ~8 LLM
                                                     # calls, 1 search, 2-3
                                                     # fetches. Cents.
    python tests/sanity_online_ch6.py --mini-run     # + tier 2: a real
                                                     # run_query(), budget-
                                                     # capped. ~30-60 LLM
                                                     # calls. Low dollars.

Tier 1 -- component sanity (each real API touched once, invariants checked):
    search    real Serper call; results tagged with formulation_id (ch. 5)
    fetch     real HTTP; source_class/identity_anchored populated (ch. 5/6);
              URL cache returns the SAME Source without a second GET
    llm       real call through call_llm; strict JSON honored; the call
              lands in a measurements DB via set_llm_logger (ch. 5)
    gate      real is_relevant -> (bool, conf); real extract_certified on
              ONE fetched page: dual extraction, validators, bootstrap
              conformal gate; invariants on whatever comes back (ch. 6)

Tier 2 -- a real (tiny) run_query(): the live analogue of
tests/test_gate_integration_offline.py. Real frontier loop, real gate,
real storage; only BUDGET shims wrap the seams (they truncate result
lists, cap fetches, sleep politely, and downgrade one-source crashes to
skips -- the Exp-1 lesson -- but always DELEGATE to the real functions).
Afterwards: DB-level invariant checks + the run funnel + per-purpose
token accounting read back from the measurements table (the ch. 5 cost
logging finally auditing a real run), + a real phi-audit on up to 3 of
the run's rejections.

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
ap.add_argument("--query",
                default=("All funding rounds raised by Hugging Face: "
                         "the amount, date, and lead investor of each round."))
ap.add_argument("--max-steps", type=int, default=2,
                help="tier 2: frontier steps (= searches issued)")
ap.add_argument("--k", type=int, default=4,
                help="tier 2: results kept per search (bounds fetch volume)")
ap.add_argument("--max-fetches", type=int, default=10,
                help="tier 2: global cap on page fetches")
ap.add_argument("--sleep", type=float, default=0.8,
                help="tier 2: seconds between fetches (politeness)")
ap.add_argument("--n-audit", type=int, default=3,
                help="tier 2: rejections to re-adjudicate in the phi-audit")
ARGS = ap.parse_args()

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webagg import config                                   # noqa: E402
from webagg import pipeline as pipe                         # noqa: E402
from webagg import fetch as fetch_mod                       # noqa: E402
from webagg import extract as extract_mod                   # noqa: E402
from webagg import audit as audit_mod                       # noqa: E402
from webagg.llm import call_llm, set_llm_logger             # noqa: E402
from webagg.search import SerperBackend                     # noqa: E402
from webagg.calibration import ConformalGate                # noqa: E402
from webagg.storage import (get_session, load_sources, load_mentions,  # noqa: E402
                            load_claims, MeasurementRow, RejectedSourceRow)

FAILURES: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def info(msg: str):
    print("  info  " + msg)


# ===========================================================================
# TIER 1: components
# ===========================================================================
def tier1() -> None:
    print("\n=== tier 1: llm wrapper (real OpenAI, logged) " + "=" * 24)
    tmp_db = f"data/runs/sanity_{uuid.uuid4().hex[:8]}.sqlite"
    session = get_session(tmp_db)
    set_llm_logger(session, "sanity_tier1")
    out = call_llm(
        system='Return STRICT JSON only: {"answer": <int>, "confidence": 0.0-1.0}',
        user="How many legs does a spider have?", purpose="sanity")
    check(isinstance(out["payload"], dict) and out["payload"].get("answer") == 8,
          f"strict-JSON payload parsed (answer={out['payload'].get('answer')})")
    check(out["input_tokens"] > 0 and out["output_tokens"] > 0,
          f"token usage returned ({out['input_tokens']} in / {out['output_tokens']} out)")
    check(out["model"] == config.MODEL_CHEAP,
          f"default model is the config cheap key ({out['model']})")
    session.commit()
    row = session.query(MeasurementRow).filter_by(metric="llm_call").one_or_none()
    check(row is not None and row.extra["purpose"] == "sanity",
          "the call landed in measurements via set_llm_logger (ch. 5 sink)")
    set_llm_logger(None, "")

    print("\n=== tier 1: search (real Serper) " + "=" * 37)
    results = SerperBackend().search("Hugging Face funding round", k=4,
                                     formulation_id="f_sanity")
    check(len(results) >= 1, f"search returned results (n={len(results)})")
    check(all(r.get("url", "").startswith("http") for r in results),
          "every result has a URL")
    check(all(r.get("formulation_id") == "f_sanity" for r in results),
          "every result tagged with the formulation that surfaced it (ch. 5)")

    print("\n=== tier 1: fetch (real HTTP) + URL cache " + "=" * 28)
    fetch_mod.clear_fetch_cache()
    src, tried = None, 0
    for r in results:
        tried += 1
        src = fetch_mod.fetch_url(r["url"], formulation_id="f_sanity")
        if src is not None:
            break
    check(src is not None, f"fetched a real page ({tried} URL(s) tried)")
    if src is None:
        return
    info(f"fetched: {src.url}  [{src.domain}]")
    check(len(src.main_text) >= 200, f"main text extracted ({len(src.main_text)} chars)")
    check(src.source_class is not None,
          f"source_class populated at fetch time (= {src.source_class})")
    check(isinstance(src.identity_anchored, bool),
          f"identity_anchored set (= {src.identity_anchored})")
    check(src.fetch_time.tzinfo is None and
          (src.publish_time is None or src.publish_time.tzinfo is None),
          "datetimes UTC-naive (project convention)")
    t0 = time.time()
    again = fetch_mod.fetch_url(str(src.url), formulation_id="f_other")
    check(again is src and (time.time() - t0) < 0.05,
          "second fetch_url = cache hit: same Source object, no second GET")
    check(again.formulation_id == "f_sanity",
          "cached Source keeps first-discoverer attribution")

    print("\n=== tier 1: the four-stage gate on that page (real LLM) " + "=" * 13)
    ok, conf = extract_mod.is_relevant(src, ARGS.query)
    check(isinstance(ok, bool) and 0.0 <= conf <= 1.0,
          f"is_relevant -> (bool, conf) = ({ok}, {conf:.2f})")
    if not ok:
        info("page judged not relevant; gate invariants exercised in tier 2")
        return
    gate = ConformalGate(delta_E=config.DELTA_E)        # unfitted: bootstrap
    mentions, claims, gi = extract_mod.extract_certified(src, ARGS.query, gate=gate)
    info(f"gate counters: {gi}")
    check(gi["agreed"] == len(mentions) + gi["validator_rejects"] + gi["gate_abstains"],
          "counter consistency: agreed = accepted + validator_rejects + abstains")
    check(gi["n_a"] == gi["agreed"] + gi["disagreed"],
          "counter consistency: n_a = agreed + disagreed")
    for m in mentions:
        check(m.accepted and m.extractor_id == "A",
              f"accepted mention well-formed ({m.attribute}={m.value!r})")
        check("gate_uncalibrated" in m.validator_flags,
              "bootstrap gate honestly flagged (no calibration file yet)")
        check(0.0 <= m.self_conf <= 1.0 and bool(m.passage),
              "self_conf in range and passage present (provenance)")
        if m.attribute in {"amount", "post_money", "valuation", "raised"}:
            check(m.currency is not None and m.value_num is not None,
                  f"money mention typed: {m.value!r} -> {m.value_num} {m.currency}")
    for c in claims:
        check(c.functional in ("SUM", "COUNT") and ":CLAIM:" in c.claim_id,
              f"claim well-formed ({c.functional} {c.value_num} '{c.scope}')")
    if not mentions:
        info("0 accepted mentions on this page (possible: dual extraction is "
             "strict) -- tier 2 checks acceptance across many pages")


# ===========================================================================
# TIER 2: budget-capped real run_query()
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
                (".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".doc", ".docx")):
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
    Delegates to the REAL post-ch.6 function."""
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
    run_id = f"sanity_live_{uuid.uuid4().hex[:8]}"
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
                                                         "agreed": 0, "disagreed": 0,
                                                         "b_only": 0,
                                                         "validator_rejects": 0,
                                                         "gate_abstains": 0}),
                                               "extract_certified")
        # keep the frontier from ballooning past the budget window
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

    # ---- DB invariants (the same contract the offline integration test locks)
    sources = {s.source_id: s for s in load_sources(session)}
    all_m = load_mentions(session)
    acc = load_mentions(session, accepted_only=True)
    claims = load_claims(session)
    rej = session.query(RejectedSourceRow).all()
    rows = session.query(MeasurementRow).filter_by(run_id=run_id).all()

    check(len(sources) >= 1, f"sources persisted (n={len(sources)})")
    check(all(s.source_class is not None for s in sources.values()),
          "every stored source carries a source_class")
    check(all(m.source_id in sources for m in all_m),
          "provenance FK: every mention's source is stored")
    check(all(m.accepted for m in acc) and
          all("gate_uncalibrated" in m.validator_flags for m in acc),
          f"accepted mentions gated + honestly flagged (n={len(acc)})")
    money = [m for m in acc if m.attribute in
             {"amount", "post_money", "valuation", "raised"}]
    check(all(m.currency and m.value_num is not None for m in money),
          f"all accepted money mentions typed (n={len(money)})")
    for c in claims:
        check(c.functional in ("SUM", "COUNT"),
              f"claim functional legal ({c.functional} {c.value_num})")
    info(f"rejected sources logged: {len(rej)} "
         f"(scores: {sorted(round(r.rejection_score or 0, 2) for r in rej)[:6]})")
    check(all(r.main_text for r in rej),
          "rejections keep main_text (phi-audit raw material)")

    metrics = {r.metric for r in rows}
    check("U_hat" in metrics, "U_hat logged per step (frontier alive)")
    check({"extract_agreed", "extract_abstained"} <= metrics or not acc,
          "gate outcome measurements present")
    llm_rows = [r for r in rows if r.metric == "llm_call"]
    purposes = {}
    tok = {}
    for r in llm_rows:
        p = r.extra["purpose"]
        purposes[p] = purposes.get(p, 0) + 1
        tok[p] = tok.get(p, 0) + int(r.value)
    check(purposes.get("extraction_A", 0) == purposes.get("extraction_B", 0),
          f"dual extraction balanced (A={purposes.get('extraction_A', 0)}, "
          f"B={purposes.get('extraction_B', 0)})")
    check(purposes.get("relevance", 0) >= len(sources) + len(rej),
          "every fetched page got a relevance verdict")

    # ---- provenance walk on the most-corroborated value, if any ----
    if acc:
        from collections import defaultdict
        from webagg.corroboration import corroborate
        groups = defaultdict(list)
        for m in acc:
            if m.value_num is not None:
                groups[(m.attribute, m.value_num)].append(m)
        key, grp = max(groups.items(), key=lambda kv: len(kv[1]), default=(None, []))
        if grp:
            cv = corroborate({str(key[1]): grp}, sources)
            urls = {str(sources[m.source_id].url) for m in grp
                    if m.mention_id in cv.supporting_mention_ids}
            check(len(cv.supporting_mention_ids) == len(grp) and len(urls) >= 1,
                  f"provenance walk: {key[0]}={key[1]:,.0f} "
                  f"(nu={cv.nu}, {len(urls)} url(s))")

    # ---- real phi-audit on a few of this run's rejections ----
    if rej:
        rho = audit_mod.phi_fn_upper(session, ARGS.query,
                                     n_audit=min(ARGS.n_audit, len(rej)))
        n_aud = session.query(RejectedSourceRow).filter_by(audited=True).count()
        check(0.0 < rho <= 1.0,
              f"phi-audit ran on {n_aud} rejection(s): rho_bar_phi = {rho:.3f} "
              "(small n -> weak bound, as it should be)")

    # ---- funnel + cost, read back from measurements (ch. 5 paying off) ----
    print("\n=== run funnel + cost (from the measurements table) " + "=" * 17)
    print(f"  sources {len(sources)}  rejected {len(rej)}  mentions "
          f"{len(all_m)} total / {len(acc)} accepted  claims {len(claims)}")
    agreed_rows = [r for r in rows if r.metric == "extract_agreed"]
    if agreed_rows:
        vr = sum(r.extra["validator_rejects"] for r in agreed_rows)
        dis = sum(r.extra["disagreed"] for r in agreed_rows)
        ga = sum(r.extra["gate_abstains"] for r in agreed_rows)
        print(f"  gate: agreed {sum(int(r.value) for r in agreed_rows)}, "
              f"A/B disagreed {dis}, validator rejects {vr}, gate abstains {ga}")
    for p in sorted(purposes):
        print(f"  llm[{p:<14}] calls {purposes[p]:>3}  tokens {tok[p]:>8,}")
    print(f"  total llm calls {len(llm_rows)}, tokens {sum(tok.values()):,}"
          f"  |  wall {elapsed:.0f}s  |  DB data/runs/{run_id}.sqlite")


# ===========================================================================
if __name__ == "__main__":
    tier1()
    if ARGS.mini_run:
        tier2()
    print(f"\n{len(FAILURES)} CHECK(S) FAILED" if FAILURES else "\nALL CHECKS PASSED")
    sys.exit(1 if FAILURES else 0)
