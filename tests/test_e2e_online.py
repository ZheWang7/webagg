"""Online end-to-end pipeline test -- impl guide Sec. 11, design doc Sec. 7.1.

Exercises the FULL wired pipeline against REAL APIs (Serper + gpt-5-nano via
call_llm), in the order the design doc composes it:

    FrontierDiscovery (Sec. 2, Thm 1)  ->  entity resolution (Sec. 5, Thm 4)
    ->  fragmentation classification (Sec. 6, Lemma 1)
    ->  corroboration per attribute, guarded for fragmenting attrs (Sec. 3 + 6.9)
    ->  f_Q over distinct resolved records  ->  3-part interval (Corollary 2)

Run from the repo root (prompts/ and .env must be in the cwd):

    python scripts/test_e2e_online.py --offline      # wiring check: no keys, no cost. RUN FIRST.
    python scripts/test_e2e_online.py                # live smoke run (tight default budget)
    python scripts/test_e2e_online.py --query "..." --attrs amount,date --max-steps 10

Why a harness instead of just calling end_to_end():
  * run_query() has NO budget caps -- a live run can burn steps/fetches
    indefinitely. We cap searches, results-per-search, and total fetches.
  * one bad page must not kill a paid run. The Exp-1 crash (gpt-5-nano
    exhausting max_completion_tokens on hidden reasoning -> ValueError ->
    tenacity RetryError re-raised after 3 tries) aborted a 1.5h run. Here,
    relevance/extract failures SKIP the source and are tallied, never raised.
  * cost telemetry: every LLM call in the run is routed through one counter
    (calls, tokens, ~$), and the report prints the discovery funnel.
  * ER without torch: entity_resolution.py imports sentence_transformers at
    module top (despite the embedder() docstring saying it is deferred), so
    the default here is a lightweight name-similarity cluster_fn injected via
    resolve_and_aggregate's cluster_fn seam. --matcher real uses the full
    stack if you have sentence-transformers installed.

The injection style mirrors the repo's own seams (cluster_fn, ER adjudicator,
schema relevance_fn): run_query has no deps parameter, so we patch the
pipeline module's globals (SerperBackend, fetch_url, is_relevant,
extract_mentions, seed_formulations, propose_followups) and restore them.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
import types
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI first: in --offline mode we must plant dummy keys BEFORE importing
# webagg, because webagg.llm builds BOTH API clients at import time
# (os.environ["ANTHROPIC_API_KEY"] / ["OPENAI_API_KEY"] raise KeyError if unset).
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="deterministic mock world: validates wiring + invariants, zero cost")
    ap.add_argument("--query",
                    default=("All funding rounds raised by Hugging Face: "
                             "the amount, date, and lead investor of each round."))
    ap.add_argument("--attrs", default="amount,date,lead_investor",
                    help="comma-separated query attributes A (Def. 1)")
    ap.add_argument("--aggregate-attr", default="amount",
                    help="attribute f_Q sums over distinct resolved records")
    ap.add_argument("--eps", type=float, default=0.9,
                    help="completeness slack; loose default so the Def. 7 stop "
                         "rule has a chance to fire inside a small budget")
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=20,
                    help="hard cap on frontier steps (= max searches issued)")
    ap.add_argument("--k", type=int, default=5,
                    help="search results kept per formulation (run_query asks "
                         "for 10; we truncate to bound fetch volume)")
    ap.add_argument("--max-fetches", type=int, default=50,
                    help="global cap on page fetch attempts across the run")
    ap.add_argument("--sleep", type=float, default=0.7,
                    help="seconds between fetches (politeness, impl Sec. 2.4)")
    ap.add_argument("--matcher", choices=["light", "real"], default="light",
                    help="light = torch-free name-similarity cluster_fn; "
                         "real = full Sec. 5 blocker+matcher+LLM band")
    ap.add_argument("--run-id", default=None)
    return ap.parse_args()


ARGS = parse_args()

if ARGS.offline:
    # dummies so `import webagg.llm` succeeds; nothing is ever called offline
    for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "SERPER_API_KEY"):
        os.environ.setdefault(_k, "offline-dummy")

if not Path("prompts/relevance.txt").exists():
    sys.exit("Run this from the repo root: prompts/ not found in the cwd "
             "(extract.py and this harness read prompts/*.txt relatively).")

# make `python scripts/test_e2e_online.py` importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# webagg imports AFTER the env guard --------------------------------------
from webagg import corroboration as corr_mod            # noqa: E402
from webagg import pipeline as pipe                     # noqa: E402
from webagg.storage import MeasurementRow               # noqa: E402
from webagg.storage import load_mentions, load_sources  # noqa: E402
from webagg.type_defs import Mention, Source            # noqa: E402

if not ARGS.offline:
    missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Live mode needs {missing} in .env "
                 "(ANTHROPIC_API_KEY must also be set for webagg.llm to import).")


# ===========================================================================
# 1.  BUDGET + LLM ACCOUNTING  (every live LLM call flows through _llm)
# ===========================================================================

@dataclass
class Budget:
    max_searches: int
    max_fetches: int
    k: int
    sleep: float
    searches: int = 0
    fetches: int = 0
    fetch_fail: int = 0
    relevant_pass: int = 0
    relevant_fail: int = 0          # relevance calls that errored -> source skipped
    extract_fail: int = 0           # extraction calls that errored -> source skipped
    mentions: int = 0
    llm_calls: int = 0
    in_tok: int = 0
    out_tok: int = 0
    skipped: list = field(default_factory=list)   # (url, reason) audit trail
    seen_urls: set = field(default_factory=set)   # URL dedup across steps


BUDGET = Budget(max_searches=ARGS.max_steps, max_fetches=ARGS.max_fetches,
                k=ARGS.k, sleep=ARGS.sleep)

# gpt-5-nano list price per 1M tokens (approx; update if OpenAI reprices)
_PRICE_IN, _PRICE_OUT = 0.05, 0.40


def _llm(*, system: str, user: str, max_tokens: int = 4096) -> dict:
    """Counted call into the repo's single-API wrapper (webagg.llm.call_llm,
    gpt-5-nano). Lazy import so --offline never touches webagg.llm's clients."""
    from webagg.llm import call_llm
    out = call_llm(system=system, user=user, max_tokens=max_tokens)
    BUDGET.llm_calls += 1
    BUDGET.in_tok += out.get("input_tokens", 0)
    BUDGET.out_tok += out.get("output_tokens", 0)
    return out["payload"]


# ===========================================================================
# 2.  LIVE WRAPPERS  (patched over webagg.pipeline's globals)
# ===========================================================================
# Each wrapper = repo behavior + (a) budget enforcement, (b) failure isolation:
# an exception on ONE source downgrades to a skip, so the run's already-paid
# work survives (the Exp-1 failure mode).

_NONCONTENT = (".pdf", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".doc", ".docx")


class BudgetedSearch:
    """Wraps SerperBackend: enforces max_searches and truncates to k results.
    run_query hardcodes k=10 in its call; we ignore that and apply our own."""

    def __init__(self):
        from webagg.search import SerperBackend
        self._real = SerperBackend()

    def search(self, query: str, k: int = 10) -> list[dict]:
        if BUDGET.searches >= BUDGET.max_searches:
            return []                       # frontier keeps looping cheaply until max_steps
        BUDGET.searches += 1
        try:
            return self._real.search(query, k=BUDGET.k)[:BUDGET.k]
        except Exception as e:              # one dead search != dead run
            BUDGET.skipped.append((f"search:{query[:60]}", repr(e)))
            return []


def budgeted_fetch(url: str, formulation_id: str):
    """fetch_url + fetch budget + politeness sleep + non-content filter
    (impl Sec. 6.4: 'search backends return non-content URLs'). fetch_url's
    tenacity retry re-raises connect errors after 3 tries -- catch, skip."""
    if BUDGET.fetches >= BUDGET.max_fetches:
        return None
    if url in BUDGET.seen_urls:
        # run_query has no URL cache (impl Sec. 5.3 says it is essential and it
        # was never built): two searches returning the same URL would create two
        # Source rows with EQUAL publish_time -> no derivation edge either way
        # (Def. 8 needs strict ordering) -> the page counts as TWO independent
        # witnesses, inflating nu and belief. Self-echo defeating Theorem 2.
        # Deduping here keeps nu honest and skips the duplicate's LLM calls.
        BUDGET.skipped.append((url, "duplicate URL (already fetched this run)"))
        return None
    BUDGET.seen_urls.add(url)
    if url.lower().split("?")[0].endswith(_NONCONTENT):
        BUDGET.skipped.append((url, "non-content extension"))
        return None
    BUDGET.fetches += 1
    time.sleep(BUDGET.sleep)                # <= ~1 req/s (impl Sec. 2.4 politeness)
    try:
        from webagg.fetch import fetch_url
        return fetch_url(url, formulation_id=formulation_id)
    except Exception as e:
        BUDGET.fetch_fail += 1
        BUDGET.skipped.append((url, f"fetch:{type(e).__name__}"))
        return None


def hardened_is_relevant(source: Source, query: str) -> bool:
    """Repo's relevance predicate (phi, Def. 1) with input capped at 6k chars
    and errors treated as 'not relevant' (conservative skip, never a crash)."""
    sysp = open("prompts/relevance.txt").read()
    user = f"QUERY:\n{query}\n\nDOCUMENT:\n{source.main_text[:6000]}"
    try:
        ok = bool(_llm(system=sysp, user=user).get("relevant"))
    except Exception as e:
        BUDGET.relevant_fail += 1
        BUDGET.skipped.append((str(source.url), f"relevance:{type(e).__name__}"))
        return False
    if ok:
        BUDGET.relevant_pass += 1
    return ok


def hardened_extract(source: Source, query: str) -> list[Mention]:
    """Repo's extract_mentions with the Exp-1 fixes baked in:
       max_tokens=8000 (room for output after gpt-5-nano's hidden reasoning)
       and input capped at 6000 chars; one lean retry at 3000 chars; on final
       failure the SOURCE is skipped, the run continues."""
    sysp = open("prompts/extract.txt").read()
    payload = None
    for max_chars in (6000, 3000):
        user = (f"QUERY:\n{query}\n\nDOCUMENT (id={source.source_id}):\n"
                f"{source.main_text[:max_chars]}")
        try:
            payload = _llm(system=sysp, user=user, max_tokens=8000)
            break
        except Exception:
            continue
    if payload is None:
        BUDGET.extract_fail += 1
        BUDGET.skipped.append((str(source.url), "extract: failed twice"))
        return []
    out = []
    for i, m in enumerate(payload.get("mentions", [])):
        try:
            # mention_id is the PRIMARY KEY of the mentions table -- it must be
            # unique per assertion (a Mention is one "atom of provenance",
            # design Def. 3). entity+kind+value is NOT unique: this page lists
            # many funding rounds that ALL share entity="Hugging Face" and
            # kind="funding_round", and two rounds can repeat a value (several
            # have lead_investor="Not publicly disclosed"; amounts can match).
            # Those hash identically -> "UNIQUE constraint failed" on commit.
            # Fold in the position i so each extracted assertion gets its own id.
            ident = f"{m['entity_surface']}|{m['record_kind']}|{m['value']}|{i}"
            h = hashlib.sha256(ident.encode()).hexdigest()[:8]
            out.append(Mention(
                mention_id=f"{source.source_id}:{m['attribute']}:{h}",
                source_id=source.source_id,
                entity_surface=m["entity_surface"],
                record_kind=m["record_kind"],
                attribute=m["attribute"],
                value=str(m["value"]),
                passage=m.get("passage", ""),
                extracted_at=datetime.utcnow(),
            ))
        except Exception:
            continue                        # one malformed mention != dead source
    BUDGET.mentions += len(out)
    return out


def _norm_query(q: str) -> str:
    # order-free token form: catches the LLM's rephrased duplicates
    return " ".join(sorted(re.findall(r"\w+", q.lower())))


def budgeted_seed(query: str):
    from webagg.frontier import Formulation
    sysp = ("Propose 5 diverse initial search formulations for the user's query. "
            'Return JSON: {"formulations":[{"query":"...","expected_yield":1-10}]}')
    out = _llm(system=sysp, user=query)
    fs, seen = [], set()
    for f in out.get("formulations", [])[:5]:
        n = _norm_query(f["query"])
        if n in seen:
            continue
        seen.add(n)
        fs.append(Formulation(formulation_id=str(uuid.uuid4())[:8],
                              query=f["query"],
                              expected_yield=float(f["expected_yield"])))
    return fs


def budgeted_followups(record_kind: str, entity_surface: str, already_tried: list[str]):
    """Repo's propose_followups + normalized dedup vs. already_tried + a cap
    of 3 per record. Uncapped, the frontier balloons and the stop rule can
    never see it exhausted inside a small budget (impl Sec. 6.4 pitfall 1)."""
    from webagg.frontier import Formulation
    sysp = open("prompts/propose_followups.txt").read()
    user = (f"record_kind: {record_kind}\nentity: {entity_surface}\n"
            f"already_tried: {already_tried[-20:]}")
    try:
        out = _llm(system=sysp, user=user)
    except Exception:
        return []                           # frontier growth is optional; discovery isn't
    tried = {_norm_query(q) for q in already_tried}
    fs = []
    for f in out.get("formulations", []):
        n = _norm_query(f.get("query", ""))
        if not n or n in tried:
            continue
        tried.add(n)
        fs.append(Formulation(formulation_id=str(uuid.uuid4())[:8],
                              query=f["query"],
                              expected_yield=float(f.get("expected_yield", 1))))
        if len(fs) == 3:
            break
    return fs


# ===========================================================================
# 3.  LIGHTWEIGHT ENTITY RESOLUTION  (default; --matcher real for the full stack)
# ===========================================================================

_SUFFIX = {"inc", "corp", "corporation", "llc", "ltd", "limited", "co", "company"}


def _norm_name(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    toks = [t for t in re.findall(r"\w+", s) if t not in _SUFFIX]
    return " ".join(toks)


def light_cluster_fn(mentions, source_lookup) -> dict[str, str]:
    """Torch-free stand-in for Sec. 5: prefix blocking (Def. 12) + a strong
    name-similarity merge threshold (the theta >= tau_plus channel only; no
    embedding features, no LLM escalation band -> zero cost, deterministic)
    + union-find for the transitive closure. Merges 'Acme Corp'/'Acme, Inc.',
    keeps 'Bolt Logistics' apart. Injected via resolve_and_aggregate's
    cluster_fn seam, exactly like tests/test_end_to_end_offline.py."""
    from rapidfuzz.fuzz import token_set_ratio
    parent = {m.mention_id: m.mention_id for m in mentions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    blocks = defaultdict(list)               # blocking: only within-block pairs
    for m in mentions:
        n = _norm_name(m.entity_surface)
        blocks[re.sub(r"\s", "", n)[:4] or n].append((m.mention_id, n))
    for members in blocks.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (ida, na), (idb, nb) = members[i], members[j]
                if na == nb or token_set_ratio(na, nb) >= 88:   # tau_plus analogue
                    union(ida, idb)

    roots, out = {}, {}
    for m in mentions:
        r = find(m.mention_id)
        if r not in roots:
            roots[r] = f"ent_{len(roots):05d}"
        out[m.mention_id] = roots[r]
    return out


# ===========================================================================
# 4.  KNOWN-BUG PATCH: reliability() vs "www." domains
# ===========================================================================
# corroboration.reliability() looks priors up by EXACT domain, but fetch_url
# stores urlparse(url).netloc -- so a live "www.sec.gov" page silently gets the
# 0.50 default instead of 0.97. fragmentation.classify() already strips "www."
# for the same reason; reliability() doesn't. Patched here so live beliefs are
# honest; the repo fix is one line at the top of reliability():
#     d = source.domain.lower();  d = d[4:] if d.startswith("www.") else d

_orig_reliability = corr_mod.reliability


def _reliability_www(source):
    d = source.domain.lower()
    if d.startswith("www."):
        d = d[4:]
    return _orig_reliability(types.SimpleNamespace(domain=d))


# ===========================================================================
# 5.  OFFLINE MOCK WORLD  (validates the same wiring with zero keys/cost)
# ===========================================================================
# Built so the run exercises every stage AND the Def. 7 stop rule fires:
#   step 1  q1 -> sec + pr + echo-of-pr + wrong-blog + bolt-news
#   step 2  q2 -> linkedin (employees, never NAMES Acme -> 6.9 guard)
#                 + crunchbase (2nd independent Bolt witness)
#   step 3  q3 (followup) -> re-serves the bolt page: 0 new records
#   after step 3: r_t = 0, frontier exhausted  =>  U_hat = 0 < eps/2  =>  STOP.

class MockWorld:
    def __init__(self):
        self.n_fetch = 0
        d0 = datetime(2025, 1, 1, 12, 0, 0)
        t = lambda day: d0 + timedelta(days=day)
        PR = ("Acme Corp today announced it has closed a forty million dollar "
              "Series B financing round led by a16z")

        def S(sid, dom, day, text):
            return dict(sid=sid, dom=dom, pub=t(day), text=text)

        self.pages = {
            "https://sec.gov/formd/acme":       S("sec1", "sec.gov", 8,
                "Form D filing for Acme Corp. Total amount sold: $40,000,000. "
                "Date of first sale: 2025-01-02."),
            "https://prnewswire.com/acme":      S("pr1", "prnewswire.com", 3, PR),
            "https://blog1.example/acme":       S("blog1", "blog1.example", 4, PR),  # echo of pr1
            "https://blog2.example/acme":       S("blog2", "blog2.example", 5,
                "Sources say Acme, Inc. actually raised $42M in the round."),
            "https://boltnews.example/bolt":    S("bolt1", "boltnews.example", 5,
                "Bolt Logistics raises $12M seed round, closing January 5."),
            "https://www.linkedin.com/acme":    S("li1", "www.linkedin.com", 6,
                "This fast-growing robotics startup now employs 87 people."),   # no 'Acme'!
            "https://crunchbase.com/bolt":      S("bolt2", "crunchbase.com", 7,
                "Bolt Logistics | Total Funding Amount: $12M."),
        }
        M = lambda attr, val, surf, url: dict(attribute=attr, value=val,
                                              entity_surface=surf,
                                              passage=self.pages[url]["text"][:80])
        self.extractions = {   # url -> mention specs (record_kind fixed below)
            "https://sec.gov/formd/acme":    [M("amount", "$40M", "Acme Corp", "https://sec.gov/formd/acme"),
                                              M("date", "2025-01-02", "Acme Corp", "https://sec.gov/formd/acme")],
            "https://prnewswire.com/acme":   [M("amount", "40 million USD", "Acme, Inc.", "https://prnewswire.com/acme"),
                                              M("date", "2025-01-02", "Acme, Inc.", "https://prnewswire.com/acme")],
            "https://blog1.example/acme":    [M("amount", "$0.04B", "Acme, Inc.", "https://blog1.example/acme")],
            "https://blog2.example/acme":    [M("amount", "$42M", "Acme, Inc.", "https://blog2.example/acme")],
            "https://boltnews.example/bolt": [M("amount", "$12M", "Bolt Logistics", "https://boltnews.example/bolt"),
                                              M("date", "2025-01-05", "Bolt Logistics", "https://boltnews.example/bolt")],
            "https://www.linkedin.com/acme": [M("employees", "87", "Acme Corp", "https://www.linkedin.com/acme")],
            "https://crunchbase.com/bolt":   [M("amount", "$12M", "Bolt Logistics", "https://crunchbase.com/bolt")],
        }
        self.results = {
            "q1": ["https://sec.gov/formd/acme", "https://prnewswire.com/acme",
                   "https://blog1.example/acme", "https://blog2.example/acme",
                   "https://boltnews.example/bolt"],
            "q2": ["https://www.linkedin.com/acme", "https://crunchbase.com/bolt"],
            "q3": ["https://boltnews.example/bolt"],       # all seen: 0 new records
        }

    # --- the same seams the live wrappers occupy ---------------------------
    def search_backend(self):
        world = self

        class _MockSearch:
            def search(self, query, k=10):
                BUDGET.searches += 1
                return [{"url": u, "title": "", "snippet": ""}
                        for u in world.results.get(query, [])]
        return _MockSearch

    def fetch(self, url, formulation_id):
        p = self.pages.get(url)
        if p is None:
            return None
        if url in BUDGET.seen_urls:           # same dedup as the live wrapper
            BUDGET.skipped.append((url, "duplicate URL (already fetched this run)"))
            return None
        BUDGET.seen_urls.add(url)
        BUDGET.fetches += 1
        self.n_fetch += 1
        ft = datetime(2025, 2, 1) + timedelta(minutes=self.n_fetch)  # unique ids on refetch
        return Source(source_id=Source.make_id(url, ft), url=url,
                      domain=p["dom"], fetch_time=ft, publish_time=p["pub"],
                      title=None, main_text=p["text"], formulation_id=formulation_id)

    def relevant(self, source, query):
        BUDGET.relevant_pass += 1
        return True

    def extract(self, source, query):
        out = []
        for spec in self.extractions.get(str(source.url), []):
            ident = f"{spec['entity_surface']}|funding_round|{spec['value']}"
            h = hashlib.sha256(ident.encode()).hexdigest()[:8]
            out.append(Mention(mention_id=f"{source.source_id}:{spec['attribute']}:{h}",
                               source_id=source.source_id,
                               entity_surface=spec["entity_surface"],
                               record_kind="funding_round",
                               attribute=spec["attribute"], value=spec["value"],
                               passage=spec["passage"], extracted_at=source.fetch_time))
        BUDGET.mentions += len(out)
        return out

    def seed(self, query):
        from webagg.frontier import Formulation
        return [Formulation("f-q1", "q1", 4.0), Formulation("f-q2", "q2", 2.0)]

    def followups(self, record_kind, entity_surface, already_tried):
        from webagg.frontier import Formulation
        if "bolt" in entity_surface.lower() and "q3" not in already_tried:
            return [Formulation("f-q3", "q3", 1.0)]
        return []


# ===========================================================================
# 6.  INVARIANT CHECKS  (the actual test)
# ===========================================================================

FAILURES: list[str] = []


def check(cond: bool, msg: str):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILURES.append(msg)


def run_checks(result, session, run_id, eps_effective, offline: bool):
    records, reports = result["records"], result["reports"]
    ci = result["ci"]

    # -- discovery persisted with provenance intact (impl Sec. 1.3 rule 2) --
    sources = {s.source_id: s for s in load_sources(session)}
    mentions = load_mentions(session)
    check(len(sources) >= 1, f"discovery persisted sources (n={len(sources)})")
    check(len(mentions) >= 1, f"discovery persisted mentions (n={len(mentions)})")
    check(all(m.source_id in sources for m in mentions),
          "every mention's source_id resolves to a stored Source (provenance FK)")

    # -- stopping (Thm 1 / Def. 7) vs budget cap ----------------------------
    rows = session.query(MeasurementRow).filter_by(run_id=run_id).all()
    u_hats = [r for r in rows if r.metric == "U_hat"]
    stops = [r for r in rows if r.metric == "stop"]
    check(len(u_hats) >= 1, f"U_hat logged every step (n={len(u_hats)})")
    check(all(r.value >= 0 for r in u_hats), "U_hat trace is non-negative")
    stop_reason = ("Def. 7 stop rule fired (U<eps/2, frontier exhausted)"
                   if stops else "budget-capped (max_steps) before the rule fired")
    print(f"         stop mode: {stop_reason} after {len(u_hats)} step(s)")
    if offline:
        check(bool(stops), "offline world: the stop rule itself fired (not the cap)")

    # -- entity resolution: every mention joined (Sec. 5) -------------------
    ents = {r["entity_id"] for r in records}
    check(len(records) >= 1, f"ER + regrouping produced resolved records (n={len(records)})")
    check(len(ents) >= 1, f"distinct entities (n={len(ents)})")

    # -- fragmentation: legal case per record (Def. 16) ----------------------
    legal = {"scan_sufficient", "fragmented", "redundant", "empty"}
    check(all(r["frag_case"] in legal for r in records),
          "every record classified into a legal fragmentation case")
    frag_logs = [r for r in rows if r.metric == "frag_case"]
    check(len(frag_logs) == len(records),
          f"one frag_case measurement per record ({len(frag_logs)}/{len(records)})")

    # -- corroboration: beliefs are probabilities; nu consistent (Thm 2) ----
    for r in records:
        for attr, cv in r["attributes"].items():
            check(0.0 <= cv.belief <= 1.0,
                  f"belief in [0,1] for {r['entity_id']}/{attr} ({cv.belief:.3f})")
            check(cv.nu >= 1 and len(cv.component_sizes) == cv.nu,
                  f"nu witnesses match component list for {r['entity_id']}/{attr}")
            check(all(0.0 <= b <= 1.0 for b in cv.competing.values()),
                  f"competing beliefs in [0,1] for {r['entity_id']}/{attr}")

    # -- aggregate + 3-part interval (Corollary 2) ---------------------------
    agg = ARGS.aggregate_attr
    recomputed = 0.0
    for r in records:
        cv = r["attributes"].get(agg)
        if cv is None:
            continue
        try:
            recomputed += float(cv.value)
        except ValueError:
            pass
    check(abs(recomputed - ci["answer"]) < 1e-6,
          f"f_Q recomputes: SUM({agg}) = {ci['answer']:,.2f}")
    check(min(ci["value_term"], ci["join_term"], ci["recall_term"]) >= 0,
          "all three interval terms non-negative")
    check(abs(ci["ci_total"] - (ci["value_term"] + ci["join_term"] + ci["recall_term"])) < 1e-6,
          "ci_total = value + join + recall")
    check(abs(ci["recall_term"] - eps_effective * ci["answer"]) < 1e-6,
          "recall term = eps * SUM (Thm 1 slack)")

    # -- budgets respected ----------------------------------------------------
    check(BUDGET.searches <= BUDGET.max_searches,
          f"search budget respected ({BUDGET.searches}/{BUDGET.max_searches})")
    check(BUDGET.fetches <= BUDGET.max_fetches,
          f"fetch budget respected ({BUDGET.fetches}/{BUDGET.max_fetches})")

    # -- offline-only exact expectations (deterministic world) ---------------
    if offline:
        check(len(ents) == 2, "light ER: {Acme Corp, Acme Inc} merged; Bolt split")
        acme = next(r for r in records
                    if "40000000" == r["attributes"].get("amount", types.SimpleNamespace(value="")).value)
        bolt = next(r for r in records if r is not acme)
        check(acme["frag_case"] == "fragmented" and
              next(rep for k, rep, _ in reports if k[0] == acme["entity_id"]).fragmenting_attrs == {"employees"},
              "Acme record complementarily fragmented on 'employees' (Def. 16.2)")
        check(bolt["frag_case"] == "scan_sufficient",
              "Bolt record scan-sufficient (Def. 16.1)")
        am = acme["attributes"]["amount"]
        check(am.nu == 2, "echo collapsed: PR + copy = one witness; SEC = second (nu=2)")
        check(abs(am.belief - (1 - (1 - 0.97) * (1 - 0.5))) < 1e-9,
              "noisy-OR belief for $40M = 0.985 (sec 0.97 (+) pr-cluster 0.5)")
        check("42000000" in am.competing, "the $42M misreport tracked as competing value")
        check(abs(acme["attributes"]["employees"].belief - 0.25) < 1e-9,
              "6.9 guard halved the un-named LinkedIn employees belief (0.5 -> 0.25)")
        check(ci["answer"] == 52_000_000.0, "SUM = 40M + 12M = 52M")


# ===========================================================================
# 7.  REPORT
# ===========================================================================

def collapse_warning(result):
    """Non-failing diagnostic for the record-resolution gap (design Sec. 5.2).

    The pipeline keys a record by (entity_id, record_kind) ONLY. So every
    funding round of one entity lands in ONE record -- exactly the collapse
    you hit keying LeBron's 22 games by entity alone. corroborate() then keeps
    a single v_star per attribute, so f_Q sums ONE round, not all of them.
    We spot it: if a record's aggregate attribute corroborated 3+ DISTINCT
    numeric values, those are almost certainly different rounds folded together
    (2 values could be a mere misreport -- the Thm 2 echo case -- so we don't
    warn on that)."""
    agg = ARGS.aggregate_attr
    hits = []
    for r in result["records"]:
        cv = r["attributes"].get(agg)
        if cv is None:
            continue
        nums = set()
        for v in {cv.value, *cv.competing.keys()}:   # values are already canonicalized
            try:
                nums.add(float(v))
            except ValueError:
                pass
        if len(nums) >= 3:
            hits.append((r["entity_id"], sorted(nums)))
    if hits:
        print("\n=== WARNING: record collapse (design Sec. 5.2 not wired) "
              + "=" * 12)
        print("  One resolved record absorbed several distinct values for "
              f"'{agg}'. f_Q is summing ONE per record, so the total UNDERCOUNTS:")
        for eid, nums in hits:
            shown = ", ".join(f"{n:,.0f}" for n in nums[:6])
            print(f"    {eid}: {len(nums)} distinct {agg}s -> {{{shown}"
                  f"{', ...' if len(nums) > 6 else ''}}}")
        print("  Fix = record resolution: within an entity, split mentions into "
              "rounds\n  keyed on (amount, date) -- the funding analogue of your "
              "LeBron (date, opponent) key.")


def report(result, run_id, eps_effective, elapsed):
    ci = result["ci"]
    print("\n=== resolved records " + "=" * 47)
    for r in result["records"]:
        cv = r["attributes"].get(ARGS.aggregate_attr)
        head = f"{r['entity_id']}  {r['record_kind']:<16} [{r['frag_case']}]"
        if cv:
            head += f"  {ARGS.aggregate_attr}={cv.value}  b={cv.belief:.3f}  nu={cv.nu}"
        print("  " + head)
        for attr, v in r["attributes"].items():
            if attr != ARGS.aggregate_attr:
                print(f"      {attr:<14}= {v.value}  (b={v.belief:.3f}, nu={v.nu})")

    print("\n=== answer (Corollary 2) " + "=" * 43)
    print(f"  SUM({ARGS.aggregate_attr}) = {ci['answer']:,.2f}  over {ci['n_records']} record(s)")
    print(f"  +/- {ci['ci_total']:,.2f}   = value {ci['value_term']:,.2f}"
          f" + join {ci['join_term']:,.2f} + completeness {ci['recall_term']:,.2f}"
          f"  (eps={eps_effective})")

    print("\n=== run funnel " + "=" * 53)
    print(f"  searches {BUDGET.searches}  fetch attempts {BUDGET.fetches}"
          f" (failed {BUDGET.fetch_fail})  relevant {BUDGET.relevant_pass}"
          f"  mentions {BUDGET.mentions}")
    print(f"  skipped-not-crashed: relevance {BUDGET.relevant_fail},"
          f" extraction {BUDGET.extract_fail}")
    if BUDGET.skipped:
        print("  first skips: " + "; ".join(f"{u[:48]} [{why}]" for u, why in BUDGET.skipped[:4]))
    cost = BUDGET.in_tok / 1e6 * _PRICE_IN + BUDGET.out_tok / 1e6 * _PRICE_OUT
    print(f"  LLM: {BUDGET.llm_calls} calls, {BUDGET.in_tok} in / {BUDGET.out_tok} out tokens"
          f"  ~${cost:.4f} (gpt-5-nano list price)  |  wall {elapsed:.0f}s")
    print(f"  run DB: data/runs/{run_id}.sqlite (auditable, impl Sec. 1.3)")
    collapse_warning(result)


# ===========================================================================
# 8.  MAIN
# ===========================================================================

def main():
    offline = ARGS.offline
    run_id = ARGS.run_id or f"e2e_{'off' if offline else 'live'}_{uuid.uuid4().hex[:8]}"
    attrs = {a.strip() for a in ARGS.attrs.split(",") if a.strip()}
    if offline:
        attrs = {"amount", "date", "employees"}     # what the mock world publishes
        query = "Total funding raised by Acme and Bolt"
    else:
        query = ARGS.query

    world = MockWorld() if offline else None
    saved = {n: getattr(pipe, n) for n in
             ("SerperBackend", "fetch_url", "is_relevant", "extract_mentions",
              "seed_formulations", "propose_followups")}
    saved_rel = corr_mod.reliability
    try:
        # occupy the seams (run_query has no deps param, so patch its globals)
        pipe.SerperBackend = world.search_backend() if offline else BudgetedSearch
        pipe.fetch_url = world.fetch if offline else budgeted_fetch
        pipe.is_relevant = world.relevant if offline else hardened_is_relevant
        pipe.extract_mentions = world.extract if offline else hardened_extract
        pipe.seed_formulations = world.seed if offline else budgeted_seed
        pipe.propose_followups = world.followups if offline else budgeted_followups
        corr_mod.reliability = _reliability_www     # Sec. 4 note above

        cluster_fn = light_cluster_fn if (offline or ARGS.matcher == "light") else None
        if cluster_fn is None:
            print("[matcher=real] using the full Sec. 5 stack "
                  "(needs sentence-transformers + torch; LLM band calls are counted)")

        t0 = time.time()
        result = pipe.end_to_end(
            query, run_id=run_id, query_attributes=attrs,
            aggregate_attr=ARGS.aggregate_attr, mode="open_web",
            eps=ARGS.eps, eta=ARGS.eta, max_steps=ARGS.max_steps,
            cluster_fn=cluster_fn,
        )
        elapsed = time.time() - t0
    finally:
        for n, v in saved.items():
            setattr(pipe, n, v)
        corr_mod.reliability = saved_rel

    session = pipe.get_session(f"data/runs/{run_id}.sqlite")
    eps_effective = ARGS.eps                        # open_web mode: Thm 1 slack applies
    print(f"\n=== invariant checks ({'OFFLINE mock' if offline else 'LIVE'}) "
          + "=" * 35)
    run_checks(result, session, run_id, eps_effective, offline)
    report(result, run_id, eps_effective, elapsed)

    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
