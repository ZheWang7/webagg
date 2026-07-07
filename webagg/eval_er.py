"""
scripts/eval_er.py
==================
Experiment 3 - Entity/record-resolution error propagation & the honest confidence
interval.  Companion to webaggr-impl.pdf §13.3.

WHAT THEOREM THIS TESTS
-----------------------
In the *design document* (webaggr.pdf) the relevant results are:
  - Theorem 4 (§5.6): "bounded-error entity join" - resolution error feeds the aggregate
    with a bounded relative error split into a missed-merge channel (delta_B * v_split)
    and a false-merge channel (alpha * v_merge).
  - Corollary 2 (§5.6): the answer is reported as
        SUM_hat  +/-  ( value_term  +  join_term  +  completeness_term )
    i.e. the three error sources (value, join, completeness) made explicit.

NUMBERING DEVIATION (worth flagging): the implementation guide calls this
"Experiment 3 / Theorem 3 / Corollary 1".  Those are the guide's *internal* labels and
are off-by-one vs the design doc, where the entity-join bound is **Theorem 4** and the
honest interval is **Corollary 2** (design-doc Theorem 3 is the *schema-addressable*
result, which is Experiment 4, not this one).  Comments below use the design-doc numbers.

GOAL (guide §13.3): show the reported 3-part interval COVERS the true aggregate, and that
deliberately degrading the matcher (lowering tau_plus) WIDENS the join term of the bound
while coverage still holds.

SINGLE-ENTITY CAVEAT (important for the LeBron query)
----------------------------------------------------
The guide designed Exp 3 around the multi-entity Alphabet query, where tau_plus governs
*entity* resolution (is "Acme Corp" == "ACME"?).  The LeBron PPG query is different:

    ONE entity (LeBron James) but MANY records (22 playoff games).

So there is nothing to resolve at the *entity* level - every mention's surface is
"LeBron James".  The join that actually matters here is *record* resolution: which mention
belongs to which **game**, keyed on (date, opponent) - exactly analogous to resolving
funding rounds by (amount, date) in design §5.5.  The repo has no record-resolution layer
(its cluster_entities() keys on the player name only), so this module adds a small,
general one and lets the swept tau_plus drive it.  On a multi-entity query the same
tau_plus *also* drives entity resolution (we keep both paths; switch with `single_entity`).

A consequence to keep in mind when reading results: because each game has a *unique date*,
record resolution on this query is naturally robust - the realized answer barely moves with
tau_plus.  What the sweep then demonstrates is that the *reported* interval covers and that
its join term widens as the matcher is assumed worse.  On a query with colliding keys
(e.g. two subsidiaries sharing a name) the realized answer itself would drift, and coverage
would be the binding check.  Both behaviours are correct; this query just exercises the
"reported bound" side more than the "realized error" side.

RUNS ON ANY QUERY
-----------------
Pass a ground-truth JSON in the repo format plus a record-field spec (the key attributes
that identify a record + the single numeric attribute to SUM).  Defaults below match the
LeBron file; swap them for Alphabet/Tesla/etc.

COST: collect_and_extract() calls the REAL Serper API and the REAL extraction LLM (no
offline stub).  It writes a corpus cache to data/runs/<run_id>.corpus.json and is
idempotent - the expensive collection runs ONCE; the tau_plus sweep that follows is pure
local compute and free to re-run with different tau/eps.

Run from the REPO ROOT (so prompts/ and .env resolve):
    python scripts/eval_er.py
or import run_experiment(...) / run_sweep(...) from notebooks/03_er_error_propagation.ipynb.
"""
from __future__ import annotations

import re
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from rapidfuzz.fuzz import token_set_ratio

# ---------------------------------------------------------------------------
# Repo imports.  These are LIGHT (sqlalchemy/pydantic/networkx/rapidfuzz/datasketch)
# and power the analysis path, so they're safe at module top.  The HEAVY / networked
# bits (search, fetch, the extraction LLM, sentence-transformers) are imported lazily
# inside collect_and_extract()/_assign_entities() so the sweep can run with no torch
# and no API keys.
# ---------------------------------------------------------------------------
from webagg.type_defs import Source, Mention
from webagg.corroboration import corroborate          # design §3.4 noisy-OR; Thm 2 echo-robust

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Default record spec for the LeBron query: a game is identified by (date, opponent),
# and the attribute we SUM is points.  Override for other queries.
LEBRON_RECORD_FIELDS = {"key": ["date", "opponent"], "aggregate": "points", "entity": None}

# Matching constants -------------------------------------------------------
DATE_SCALE_DAYS = 1.5   # date-distance falloff: same day -> 1.0, ~1 day -> 0.33, >=2 days -> 0
KEY_WEIGHTS = {"date": 0.7, "opponent": 0.3}   # date dominates (it is the discriminative key)


# ===========================================================================
# 0.  small value canonicalizers
# ===========================================================================
def parse_number(s):
    """Pull a float out of a messy value: '24' -> 24.0, '$40M' -> 4.0e7, '46 pts' -> 46.0.
    Returns None if there is no number (e.g. a team name)."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip().lower().replace(",", "").replace("$", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if not m:
        return None
    x = float(m.group())
    suf = t[m.end():m.end() + 1]                       # optional magnitude suffix
    x *= {"k": 1e3, "m": 1e6, "b": 1e9}.get(suf, 1.0)
    return x


def canon_agg_value(s) -> str | None:
    """Canonical string key for the aggregate value, so echoes of the same number group
    together before corroboration (e.g. '24', '24 points', '24.0' -> '24')."""
    x = parse_number(s)
    if x is None:
        return None
    return str(int(x)) if float(x).is_integer() else repr(x)


def _try_date(v):
    """Parse a value as a date, or None.  pandas handles 'Apr 15, 2018' and '2018-04-15'."""
    try:
        d = pd.to_datetime(str(v), errors="coerce")
        return None if pd.isna(d) else d
    except Exception:
        return None


# ===========================================================================
# 1.  the RECORD matcher  (design §5.4-§5.5, adapted to games)
#     theta(rec_a, rec_b) in [0,1]; >= tau_plus => the two mentions are the same record.
#     For the LeBron query the key attrs are date+opponent; the same code handles
#     funding rounds (amount+date) because per-key similarity auto-detects type.
# ===========================================================================
def key_similarity(attr: str, a, b) -> float:
    """Per-attribute similarity, type-aware:
       - dates  -> closeness in DAYS (a clean date is a near-perfect key; different games
                   are >=2 days apart -> ~0, so they will NOT falsely merge);
       - numbers-> relative closeness;
       - else   -> lexical token_set_ratio (e.g. 'Indiana' vs 'Indiana Pacers' -> high)."""
    da, db = _try_date(a), _try_date(b)
    if da is not None and db is not None:
        days = abs((da - db).days)
        return max(0.0, 1.0 - days / DATE_SCALE_DAYS)
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        denom = max(1.0, abs(na), abs(nb))
        return max(0.0, 1.0 - abs(na - nb) / denom)
    return token_set_ratio(str(a or ""), str(b or "")) / 100.0


def record_pair_score(fields_a: dict, fields_b: dict, key_attrs, weights=None) -> float:
    """Weighted mean of per-key similarities over the record's key attributes."""
    weights = weights or {}
    num = den = 0.0
    for k in key_attrs:
        w = weights.get(k, 1.0)
        num += w * key_similarity(k, fields_a.get(k), fields_b.get(k))
        den += w
    return num / den if den else 0.0


# ===========================================================================
# 2.  ground truth
# ===========================================================================
def load_ground_truth(path, agg_attr: str) -> dict:
    """Read a ground-truth file (repo format) and compute the true SUM, record count, and
    (for single-entity per-record queries) the implied mean = SUM / N."""
    gt = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = gt["records"]
    true_sum = sum(parse_number(r["attributes"][agg_attr]) for r in recs)
    entities = {r.get("entity_canonical") for r in recs}
    return {
        "query": gt.get("query", ""),
        "records": recs,
        "true_sum": true_sum,
        "true_n": len(recs),
        "true_mean": true_sum / max(len(recs), 1),
        "entities": entities,
        "single_entity": len(entities) == 1,        # LeBron -> True
    }


# ===========================================================================
# 3.  COLLECTION  (REAL Serper + REAL extraction LLM)  -- cached, idempotent
#     We do NOT reuse run_query()'s flat per-attribute extraction: a single game-log page
#     lists 22 games, and the flat schema can't tie 'points=24' to 'date=2018-04-15'.
#     Instead we extract RECORDS (one row per game) so a game's key attrs and its aggregate
#     value stay linked - the unit record resolution + corroboration need.  Search/fetch
#     are the repo's real functions; this is genuine paid web access.
# ===========================================================================
def _record_extract_one(main_text: str, query: str, record_fields: dict) -> list[dict]:
    """One LLM call -> list of records, each a dict of the requested fields plus a verbatim
    'passage' (provenance is sacred, guide §1.3).  Uses the repo's single call_llm()."""
    from webagg.llm import call_llm                    # lazy: keeps OpenAI/Anthropic client out of import

    fields = list(record_fields["key"]) + [record_fields["aggregate"]]
    ent_attr = record_fields.get("entity")
    if ent_attr:
        fields = [ent_attr] + fields
    schema_fields = fields + ["passage"]
    system = (
        "You extract a LIST OF RECORDS from a document for a structured query. "
        "Each record is ONE logical item (here: one game / one funding round / etc.). "
        f"For each record return EXACTLY these JSON fields: {schema_fields}. "
        "Copy values as supported by the document; if a field is absent use null. "
        "'passage' is a short verbatim snippet that supports the record. "
        'Return STRICT JSON only: {"records":[{...}, ...]}. Use [] if none are present.'
    )
    user = f"QUERY:\n{query}\n\nFIELDS: {fields}\n\nDOCUMENT:\n{main_text[:12000]}"
    payload = call_llm(system=system, user=user, max_tokens=4096)["payload"]
    return payload.get("records", []) or []


def collect_and_extract(query: str, run_id: str, record_fields: dict, *,
                        entity_name: str = "", single_entity: bool = True,
                        n_results_per_query: int = 10, force: bool = False) -> dict:
    """Build (or load) the corpus for the experiment.  Writes data/runs/<run_id>.corpus.json.
    Idempotent: if the cache exists and force=False, it is loaded and NO API calls are made.

    Pipeline per the design doc's fetch/extract worker (§8): seed formulations (LLM) ->
    search (Serper) -> fetch (HTTP+trafilatura) -> relevance filter (LLM) -> record extract
    (LLM).  All real, all logged into the corpus with provenance."""
    cache = RUNS_DIR / f"{run_id}.corpus.json"
    if cache.exists() and not force:
        print(f"[collect] using cached corpus {cache.name} (no API calls)")
        return json.loads(cache.read_text(encoding="utf-8"))

    # Lazy imports of the networked/heavy pieces (only needed when we actually collect).
    from webagg.search import SerperBackend
    from webagg.fetch import fetch_url
    from webagg.extract import is_relevant
    from webagg.pipeline import seed_formulations

    agg_attr = record_fields["aggregate"]
    key_attrs = record_fields["key"]
    ent_attr = record_fields.get("entity")

    search = SerperBackend()
    formulations = seed_formulations(query)            # real LLM: diverse initial searches
    print(f"[collect] {len(formulations)} seed formulations")

    sources: dict[str, dict] = {}
    mentions: list[dict] = []
    seen_urls: set[str] = set()

    for f in formulations:
        try:
            results = search.search(f.query, k=n_results_per_query)   # real Serper (paid)
        except Exception as e:
            print(f"[collect]  search failed for {f.query!r}: {e}")
            continue
        for r in results:
            url = r["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            src = fetch_url(url, formulation_id=f.formulation_id)      # real fetch
            if src is None or not is_relevant(src, query):            # real relevance LLM
                continue
            sid = src.source_id
            sources[sid] = {
                "url": str(src.url), "domain": src.domain,
                "publish_time": src.publish_time.isoformat() if src.publish_time else None,
                "fetch_time": src.fetch_time.isoformat(),
                "main_text": (src.main_text or "")[:4000],   # keep enough for derivation edges
            }
            try:
                recs = _record_extract_one(src.main_text, query, record_fields)   # real extract LLM
            except Exception as e:
                print(f"[collect]  extract failed for {url}: {e}")
                continue
            for rec in recs:
                if rec.get(agg_attr) is None:           # need the aggregate value to count the record
                    continue
                mentions.append({
                    "source_id": sid,
                    "entity_surface": str(rec.get(ent_attr) or entity_name or "ENTITY"),
                    "fields": {k: rec.get(k) for k in key_attrs},
                    "agg_value": rec.get(agg_attr),
                    "passage": str(rec.get("passage") or "")[:600],
                })
        print(f"[collect]  after {f.query!r}: {len(sources)} sources, {len(mentions)} record-mentions")

    corpus = {
        "query": query, "record_fields": record_fields,
        "entity_name": entity_name, "single_entity": single_entity,
        "sources": sources, "mentions": mentions,
    }
    cache.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"[collect] wrote {cache.name}: {len(sources)} sources, {len(mentions)} record-mentions")
    return corpus


# ===========================================================================
# 4.  RESOLUTION  (entities -> records-within-entity)  +  CORROBORATION
# ===========================================================================
def _sources_from_corpus(corpus: dict) -> dict[str, Source]:
    """Rebuild minimal pydantic Sources so corroborate()'s provenance graph can read
    domains/timestamps/text (design §3.2 derivation edges)."""
    out = {}
    for sid, s in corpus["sources"].items():
        out[sid] = Source(
            source_id=sid,
            url=s["url"] if str(s["url"]).startswith("http") else f"https://{s['domain']}",
            domain=s["domain"],
            fetch_time=datetime.fromisoformat(s["fetch_time"]),
            publish_time=datetime.fromisoformat(s["publish_time"]) if s["publish_time"] else None,
            title=None, main_text=s.get("main_text", ""), formulation_id="exp3",
        )
    return out


def _mention_obj(m: dict, attr: str) -> Mention:
    """A repo Mention asserting this record's aggregate value, for corroborate()."""
    h = hashlib.sha256(f"{m['source_id']}|{m['agg_value']}".encode()).hexdigest()[:8]
    return Mention(
        mention_id=f"{m['source_id']}:{attr}:{h}", source_id=m["source_id"],
        entity_surface=m["entity_surface"], record_kind="record",
        attribute=attr, value=str(m["agg_value"]), passage=m["passage"],
        extracted_at=datetime.utcnow(),
    )


def _assign_entities(mentions: list[dict], single_entity: bool, tau_plus: float,
                     source_lookup: dict[str, Source]) -> list[str]:
    """Return an entity id per mention (index-aligned).
       single_entity -> everyone is 'ent_0' (LeBron); skips sentence-transformers entirely.
       else          -> repo cluster_entities() with the swept tau_plus (design §5)."""
    if single_entity:
        return ["ent_0"] * len(mentions)
    # Multi-entity path: lazily pull in the heavy ER module (sentence-transformers).
    from webagg.entity_resolution import cluster_entities, Matcher
    objs = [_mention_obj(m, "entity") for m in mentions]
    mid_to_idx = {o.mention_id: i for i, o in enumerate(objs)}
    matcher = Matcher(tau_plus=tau_plus)               # <-- tau_plus drives ER on multi-entity queries
    mid_to_ent = cluster_entities(objs, matcher, source_lookup)
    ents = [""] * len(mentions)
    for mid, ent in mid_to_ent.items():
        ents[mid_to_idx[mid]] = ent
    return ents


def _cluster_records(ms: list[dict], tau_plus: float, key_attrs, weights) -> list[list[dict]]:
    """Record resolution WITHIN one entity: build a merge graph over mentions (edge iff
    record_pair_score >= tau_plus), then connected components = distinct records
    (design §5.4 correlation-clustering, simplified to connected components for the prototype)."""
    n = len(ms)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if record_pair_score(ms[i]["fields"], ms[j]["fields"], key_attrs, weights) >= tau_plus:
                G.add_edge(i, j)
    return [[ms[i] for i in comp] for comp in nx.connected_components(G)]


def resolve(corpus: dict, tau_plus: float, key_attrs, agg_attr: str, weights):
    """Full resolution at one tau_plus: entities -> records -> one corroborated aggregate
    value per record.  Returns (resolved_records, source_lookup)."""
    mentions = corpus["mentions"]
    source_lookup = _sources_from_corpus(corpus)
    ent_of = _assign_entities(mentions, corpus.get("single_entity", True), tau_plus, source_lookup)

    by_ent: dict[str, list[dict]] = defaultdict(list)
    for i, m in enumerate(mentions):
        by_ent[ent_of[i]].append(m)

    resolved = []
    for ent, ms in by_ent.items():
        for cluster in _cluster_records(ms, tau_plus, key_attrs, weights):
            # Group this record's mentions by their (canonical) aggregate value, then let the
            # corroboration layer pick ONE value with a belief.  N echoing copies of the same
            # number collapse to one independent witness (design Thm 2), so they neither
            # inflate the SUM nor over-confidence the belief.
            by_value: dict[str, list[Mention]] = defaultdict(list)
            for m in ms_in_cluster(cluster):
                cval = canon_agg_value(m["agg_value"])
                if cval is None:
                    continue
                by_value[cval].append(_mention_obj(m, agg_attr))
            if not by_value:
                continue
            cv = corroborate(by_value, source_lookup)            # design §3.4 / Algorithm 2
            resolved.append({
                "entity": ent,
                "key": {k: cluster[0]["fields"].get(k) for k in key_attrs},
                "value": parse_number(cv.value),
                "belief": cv.belief,
                "nu": cv.nu,                                      # independent witnesses
                "n_mentions": sum(len(v) for v in by_value.values()),
                "competing": cv.competing,
            })
    return resolved, source_lookup


def ms_in_cluster(cluster):
    """Tiny helper so the loop above reads clearly (a cluster is just its list of mentions)."""
    return cluster


# ===========================================================================
# 5.  THE THREE-PART CONFIDENCE INTERVAL  (design §5.6, Corollary 2)
# ===========================================================================
def eps_er_for_tau(tau_plus: float, base: float = 0.05, ref: float = 0.9, slope: float = 0.75) -> float:
    """Map the matcher threshold to the join-error coefficient eps_ER used in the bound.
    Rationale (design §5.6, Thm 4): lowering tau_plus widens the auto-merge zone and shrinks
    the LLM-adjudicated band, so the per-pair false-merge rate alpha rises -> a larger eps_ER.
    The guide's 'degrade tau_plus to WIDEN the bound' is exactly this monotone relationship.
        tau=0.9 -> 0.05,  0.8 -> 0.125,  0.7 -> 0.20."""
    return max(base, base + slope * (ref - tau_plus))


def aggregate_with_ci(resolved: list[dict], *, eps: float, eps_er: float, z: float = 1.96) -> dict:
    """SUM the corroborated values and attach the 3-part interval (design Corollary 2):
        SUM_hat +/- ( value_term + join_term + completeness_term )
      value_term        = z * sqrt( sum  v^2 * (1-belief)^2 )     # per-attribute belief (Thm 2)
      join_term         = eps_ER * (v_split + v_merge)            # resolution error (Thm 4)
      completeness_term = eps * SUM                               # unseen-mass slack (Thm 1)
    Mirrors the guide's aggregate_with_ci, including v_split == v_merge == mean record value."""
    res = [r for r in resolved if r["value"] is not None]
    total = sum(r["value"] for r in res)
    n = len(res)
    sigma2_value = sum((r["value"] ** 2) * (1 - r["belief"]) ** 2 for r in res)
    v_bar = total / max(n, 1)                      # average per-record value at risk
    v_split = v_merge = v_bar

    value_term = z * math.sqrt(sigma2_value)
    join_term = eps_er * (v_split + v_merge)
    recall_term = eps * total
    return {
        "answer": total, "n_records": n,
        "value_term": value_term, "join_term": join_term, "recall_term": recall_term,
        "ci_total": value_term + join_term + recall_term,
    }


# ===========================================================================
# 6.  THE SWEEP  (pure compute -- free to re-run with different tau/eps)
# ===========================================================================
def run_sweep(corpus: dict, gt: dict, *, taus=(0.9, 0.8, 0.7), eps: float = 0.10,
              z: float = 1.96, record_fields: dict | None = None) -> pd.DataFrame:
    """For each tau_plus: resolve -> corroborate -> aggregate -> CI -> check coverage.
    `gt` comes from load_ground_truth().  Returns one row per tau_plus."""
    rf = record_fields or LEBRON_RECORD_FIELDS
    key_attrs, agg_attr = rf["key"], rf["aggregate"]
    true_sum = gt["true_sum"]
    rows = []
    for tau in taus:
        resolved, _ = resolve(corpus, tau, key_attrs, agg_attr, KEY_WEIGHTS)
        eps_er = eps_er_for_tau(tau)
        ci = aggregate_with_ci(resolved, eps=eps, eps_er=eps_er, z=z)
        abs_err = abs(ci["answer"] - true_sum)
        rows.append({
            "tau_plus": tau,
            "eps_ER": round(eps_er, 4),
            "answer": ci["answer"],
            "n_records": ci["n_records"],
            "true_sum": true_sum,
            "abs_err": abs_err,
            "value_term": ci["value_term"],
            "join_term": ci["join_term"],
            "recall_term": ci["recall_term"],
            "ci_total": ci["ci_total"],
            "covered": bool(abs_err <= ci["ci_total"]),       # the Exp-3 pass condition
            # convenience: per-record mean (=PPG for the LeBron query) and its implied interval
            "mean": ci["answer"] / max(ci["n_records"], 1),
            "true_mean": gt["true_mean"],
            "mean_ci": ci["ci_total"] / max(ci["n_records"], 1),
        })
    df = pd.DataFrame(rows)
    df.attrs["eps"] = eps
    return df


# ===========================================================================
# 7.  PLOT  (guide §13.3: bar = answer, error bar = CI, line = true_sum)
# ===========================================================================
def plot_experiment3(df: pd.DataFrame, *, title_suffix: str = "", out_path: str | None = None):
    """Two panels:
       (A) the guide's figure - answer per tau_plus with the CI as an error bar and the
           true total as a horizontal line (coverage = line falls within the bar's error bar);
       (B) the CI DECOMPOSITION (value / join / completeness) stacked, so the join term's
           growth as tau_plus degrades is visible even when completeness dominates."""
    import matplotlib.pyplot as plt

    true_sum = float(df["true_sum"].iloc[0])
    x = np.arange(len(df))
    labels = [f"{t:.2f}" for t in df["tau_plus"]]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A
    colors = ["#2e7d32" if c else "#c62828" for c in df["covered"]]
    axA.bar(x, df["answer"], yerr=df["ci_total"], capsize=6, color=colors, alpha=0.85,
            edgecolor="black", linewidth=0.6)
    axA.axhline(true_sum, color="black", linestyle="--", linewidth=1.3,
                label=f"true SUM = {true_sum:g}")
    for xi, (ans, cov) in enumerate(zip(df["answer"], df["covered"])):
        axA.text(xi, ans, ("covered" if cov else "MISS"), ha="center", va="bottom", fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_xlabel(r"matcher threshold $\tau_+$ (lower = worse matcher)")
    axA.set_ylabel("aggregate (SUM of points)")
    axA.set_title("Answer vs CI vs truth" + title_suffix)
    axA.legend(loc="lower left", fontsize=8)

    # Panel B
    axB.bar(x, df["value_term"], label="value", color="#1565c0")
    axB.bar(x, df["join_term"], bottom=df["value_term"], label="join (grows as $\\tau_+\\downarrow$)",
            color="#ef6c00")
    axB.bar(x, df["recall_term"], bottom=df["value_term"] + df["join_term"],
            label="completeness", color="#6a1b9a")
    axB.set_xticks(x); axB.set_xticklabels(labels)
    axB.set_xlabel(r"matcher threshold $\tau_+$")
    axB.set_ylabel("CI half-width contribution")
    axB.set_title("CI decomposition (Corollary 2)")
    axB.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[plot] saved {out_path}")
    return fig


# ===========================================================================
# 8.  one-call orchestration (collect once, then sweep + plot)
# ===========================================================================
def run_experiment(query: str, gt_path: str, *, run_id: str,
                   record_fields: dict | None = None, entity_name: str = "",
                   taus=(0.9, 0.8, 0.7), eps: float = 0.10,
                   figure_path: str | None = None, force_collect: bool = False):
    """End-to-end: real collection (cached) -> tau_plus sweep -> coverage table -> figure."""
    rf = record_fields or LEBRON_RECORD_FIELDS
    gt = load_ground_truth(gt_path, rf["aggregate"])
    corpus = collect_and_extract(query, run_id, rf, entity_name=entity_name or "",
                                 single_entity=gt["single_entity"], force=force_collect)
    df = run_sweep(corpus, gt, taus=taus, eps=eps, record_fields=rf)
    fig = plot_experiment3(df, out_path=figure_path) if figure_path else None
    return df, corpus, gt, fig


if __name__ == "__main__":
    # LeBron 2018 playoffs PPG, open-web mode, real Serper + real extraction LLM.
    Q = ("How many points did LeBron James score in each game of the 2018 NBA Playoffs, "
         "and what was his playoff points-per-game?")
    GT = REPO_ROOT / "data" / "ground_truth" / "lebron_2018_playoffs_ppg.json"
    df, corpus, gt, _ = run_experiment(
        Q, str(GT), run_id="exp3_lebron",
        record_fields=LEBRON_RECORD_FIELDS, entity_name="LeBron James",
        taus=(0.9, 0.8, 0.7), eps=0.10,
        figure_path=str(REPO_ROOT / "figures" / "exp3_er_propagation.pdf"),
    )
    pd.set_option("display.width", 140, "display.max_columns", 30)
    print("\n=== Experiment 3: ER error propagation (LeBron 2018 playoffs) ===")
    print(df.to_string(index=False))
    print(f"\ntrue SUM={gt['true_sum']:g} over {gt['true_n']} games "
          f"(true PPG={gt['true_mean']:.2f}); records collected={len(corpus['mentions'])}")
    print(f"coverage: {int(df['covered'].sum())}/{len(df)} tau values covered the truth")
