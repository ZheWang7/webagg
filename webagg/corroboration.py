"""
Section 8 -- The Corroboration Layer (SIGMOD implementation guide, ch. 8).

Runs AFTER entity resolution has grouped mentions into candidate records.
For every attribute of every record (design paper Sec. 4), four steps:

  (1) build the provenance graph G_v over ALL sources asserting ANY value
      of the attribute, and find its components (= independent origins);
  (2) partition assertions into versions via SUPERSESSION edges, and kill
      dead versions AND their derivation descendants (stale echoes);
  (3) compute the noisy-OR belief over the LIVE versions' components using
      FIXED, capped source reliabilities (no learning in the default path);
  (4) emit the forgery margin kappa alongside the adopted value -- a
      reported DIAGNOSTIC, not a guarantee term: the operator's residual
      error is absorbed by the fidelity certificate (guide Sec. 13).

Key change vs. the pre-SIGMOD version of this file: the derivation graph is
built at ATTRIBUTE level, not per value. An echo of a Form D asserts the
Form D's value -- the derivation path is what carries its death sentence
when a Form D/A supersedes it. Per-value SUBGRAPHS are then taken for the
belief computation.
"""
import math
import re

import networkx as nx
from datasketch import MinHash, MinHashLSH

from webagg.type_defs import Source, CorroboratedValue

SHINGLE_K = 5           # 5-grams of tokens for near-duplicate detection
VERBATIM_RUN = 25       # tokens of verbatim overlap that imply copying
LSH_GATE = 50           # guide 8.1: use LSH pre-clustering only when |S| > 50


# ===========================================================================
# 8.1  Derivation edges
# ===========================================================================
# Four observable signals make a directed edge s_i -> s_j ("s_j copied s_i"):
# publication-time order, shingled Jaccard above a threshold, an explicit
# citation/link, and a long verbatim run.
#
# "Do not over-tune the threshold" (guide 8.1): for short facts, text alone
# cannot distinguish a copy from an independent assertion (design paper
# remark in Sec. 4.4). The layer's real defenses are the component
# structure, the qbar cap, and the margin -- not shingle precision. So tau
# stays at 0.85 and we move on.

def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    """Set of k-token shingles ('k-grams') of the text, lowercased."""
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def shingle_jaccard(a: str, b: str) -> float:
    """Jaccard similarity of the two texts' shingle sets, in [0,1]."""
    A, B = shingles(a), shingles(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def longest_common_verbatim_tokens(a: str, b: str) -> int:
    """Length (in tokens) of the longest matching contiguous token run."""
    A = re.findall(r"\w+", a.lower())
    B = re.findall(r"\w+", b.lower())
    best = 0
    dp = [0] * (len(B) + 1)
    for ta in A:
        new = [0] * (len(B) + 1)
        for j, tb in enumerate(B, 1):
            if ta == tb:
                new[j] = dp[j - 1] + 1
                best = max(best, new[j])
        dp = new
    return best


def derivation_edge(s_i: Source, s_j: Source, tau: float = 0.85) -> bool:
    """True iff the four signals say s_j copied s_i (guide 8.1).

    Compares the sources' MAIN TEXT (not the mention passages): an echo of
    a Form D may assert the Form D's value in different local wording, but
    the page-level text is what betrays the copy.
    """
    # signal 0 (a veto): s_j cannot copy the future. Missing timestamps
    # do not veto -- the other signals still get their say.
    ti = s_i.publish_time or s_i.fetch_time
    tj = s_j.publish_time or s_j.fetch_time
    if tj and ti and tj <= ti:
        return False
    # signal 1: near-duplicate page text
    if shingle_jaccard(s_i.main_text, s_j.main_text) >= tau:
        return True
    # signal 2: explicit citation/link back to s_i
    if str(s_i.url) in (s_j.main_text or "") or s_i.domain in (s_j.main_text or ""):
        return True
    # signal 3: a long verbatim run lifted from s_i
    if longest_common_verbatim_tokens(s_i.main_text, s_j.main_text) >= VERBATIM_RUN:
        return True
    return False


def _candidate_pairs(sources: list[Source]) -> list[tuple[int, int]]:
    """Index pairs worth testing with derivation_edge.

    Below the gate: all O(n^2) pairs (n is small, exactness is free).
    Above the gate (guide 8.1: only when |S| > 50): MinHash-LSH buckets the
    near-duplicate candidates so we skip obviously-unrelated pairs. Note the
    citation signal can still be missed for pairs LSH prunes -- accepted
    trade at this scale; the component structure absorbs the slack.
    """
    n = len(sources)
    if n <= LSH_GATE:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    lsh = MinHashLSH(threshold=0.5, num_perm=64)   # loose threshold: recall over precision
    mhs = []
    for i, s in enumerate(sources):
        mh = MinHash(num_perm=64)
        for sh in shingles(s.main_text):
            mh.update(sh.encode())
        mhs.append(mh)
        lsh.insert(str(i), mh)
    pairs = set()
    for i in range(n):
        for key in lsh.query(mhs[i]):
            j = int(key)
            if i != j:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def build_attribute_graph(mentions_by_value: dict[str, list],
                          source_lookup: dict[str, Source]) -> nx.DiGraph:
    """Derivation graph over ALL sources asserting ANY value of the attribute.

    Guide 8.2, closing note: build deriv_G over all sources of the attribute
    (an echo of the Form D asserts the Form D's value; the derivation path
    carries its death sentence), then take per-value subgraphs for belief.
    """
    G = nx.DiGraph()
    src_ids = {m.source_id for ms in mentions_by_value.values() for m in ms}
    srcs = [source_lookup[sid] for sid in src_ids]
    for s in srcs:
        G.add_node(s.source_id)
    for i, j in _candidate_pairs(srcs):
        a, b = srcs[i], srcs[j]
        # test both directions; the temporal veto inside derivation_edge
        # decides which way (if either) the copy went
        if derivation_edge(a, b):
            G.add_edge(a.source_id, b.source_id)
        elif derivation_edge(b, a):
            G.add_edge(b.source_id, a.source_id)
    return G


# ===========================================================================
# 8.2  Supersession: authority chains and version partition
# ===========================================================================
# Edges are STRUCTURAL only (design paper Def. in Sec. 4.3) -- never recency
# or popularity. Two structural signals:
#   (a) registry chains: a later amendment doc_type in the same
#       authority_chain_id (e.g. Form D/A after Form D);
#   (b) publisher self-corrections: same domain, later time, correction
#       language near the top of the page.

CORRECTION_RX = re.compile(
    r"\b(correction|corrected|updated to reflect|"
    r"an earlier version|amends|amended)\b", re.I)

_AMENDMENT_RX = re.compile(r"(/A\b|amend|updated)", re.I)


def is_amendment(doc_type: str | None) -> bool:
    """True for amendment-style doc types: 'Form D/A', '10-K/A', 'updated'."""
    return bool(doc_type and _AMENDMENT_RX.search(doc_type))


def _later(b: Source, a: Source) -> bool:
    """b strictly later than a (publish_time, falling back to fetch_time)."""
    ta = a.publish_time or a.fetch_time
    tb = b.publish_time or b.fetch_time
    return bool(ta and tb and tb > ta)


def supersession_edges(sources: list[Source]) -> list[tuple[str, str, str]]:
    """All (old_source_id, new_source_id, reason) supersession edges."""
    edges = []
    # (a) registry authority chains: sort each chain by time; every
    # (older, newer) adjacent pair where the newer doc is an amendment
    # supersedes the older
    chains: dict[str, list[Source]] = {}
    for s in sources:
        if s.authority_chain_id:
            chains.setdefault(s.authority_chain_id, []).append(s)
    for docs in chains.values():
        if len(docs) < 2:
            continue
        docs = sorted(docs, key=lambda s: s.publish_time or s.fetch_time)
        for older, newer in zip(docs, docs[1:]):
            if is_amendment(newer.doc_type):
                edges.append((older.source_id, newer.source_id, "form_amendment"))
    # (b) publisher self-corrections: same domain, later, correction wording
    # in the first 2000 chars (headlines/edit notes live at the top)
    for a in sources:
        for b in sources:
            if (a.source_id != b.source_id and a.domain == b.domain
                    and _later(b, a)
                    and CORRECTION_RX.search((b.main_text or "")[:2000])):
                edges.append((a.source_id, b.source_id, "self_correction"))
    return edges


def live_values(mentions_by_value: dict[str, list],
                source_lookup: dict[str, Source],
                deriv_G: nx.DiGraph,
                qtable: "QTable",
                q_min: float = 0.5):
    """Partition values into live vs. dead; return (live, excluded, dead_srcs).

    Dead source = a superseded chain doc (provided the SUCCESSOR's origin has
    reliability >= q_min -- we don't let a junk page kill a registry filing)
    OR any derivation DESCENDANT of one (echoes inherit the version they
    copied). A value is live if ANY of its assertions is live: stale echoes
    are DISQUALIFIED, never outvoted.

    Adaptation vs. the guide's pseudocode: the guide groups a flat mention
    list by m.value_canon; in our pipeline the canonical value is already
    the KEY of mentions_by_value (canonicalize_value() runs upstream), so we
    take the dict directly. We also return dead_srcs so corroborate() can
    drop dead assertions inside otherwise-live values -- a small tightening
    beyond the guide's verbatim code, documented as a deliberate deviation.
    """
    sources = list(source_lookup.values())
    sup = supersession_edges(sources)
    # a superseded doc dies only if its successor is itself credible
    dead = {old for old, new, _ in sup
            if qtable.q(source_lookup[new]) >= q_min}
    # propagate the death sentence down the derivation graph: everything
    # that copied a dead doc (transitively) is a stale echo
    dead_srcs = set(dead)
    for d in dead:
        if d in deriv_G:
            dead_srcs |= nx.descendants(deriv_G, d)
    live, excluded = set(), {}
    for v, ms in mentions_by_value.items():
        alive = [m for m in ms if m.source_id not in dead_srcs]
        excluded[v] = len(ms) - len(alive)   # count of mentions, auditable
        if alive:
            live.add(v)
    return live, excluded, dead_srcs


# ===========================================================================
# 8.3  Belief over live versions, with FIXED capped reliabilities
# ===========================================================================
# The reliability table holds fixed SOURCE-TYPE priors; an origin without an
# identity anchor is capped at qbar. There is NO learning in the default
# path (design choice the guide tracks; optional refinements live in 8.5,
# not implemented here). No core theorem depends on the specific q values;
# the cap is a SECURITY control against forged origins.

CLASS_PRIOR = {"registry": 0.95, "vendor": 0.80, "news": 0.60,
               "blog": 0.40, "other": 0.50}


class QTable:
    """Default reliability model: fixed class priors + qbar cap. No EM.

    (Optional refinements -- learning q from certified strata, or EM --
    are guide Sec. 8.5 and deliberately NOT implemented yet.)
    """

    def __init__(self, qbar: float = 0.30):
        self.qbar = qbar    # adversarial cap: max reliability without an identity anchor

    def q(self, source: Source) -> float:
        """Reliability prior for one source, capped if unanchored."""
        base = CLASS_PRIOR.get(source.source_class or "other", 0.50)
        # identity_anchored = registry / known publisher / entity's own
        # domain. Anything else could be a forged origin -> cap at qbar.
        return base if source.identity_anchored else min(base, self.qbar)


def per_value_subgraph(deriv_G: nx.DiGraph, mentions: list) -> nx.DiGraph:
    """Subgraph of the attribute graph induced by this value's sources."""
    return deriv_G.subgraph({m.source_id for m in mentions})


def margin(b_star: float, b2: float, qbar: float) -> int:
    """Forgery margin kappa (design paper Sec. 4 margin theorem / Prop. 1).

    How many forged qbar-reliability independent witnesses an adversary
    must add behind the runner-up before it overtakes the adopted value.
    Each forged witness multiplies the runner-up's (1 - belief) by
    (1 - qbar), so kappa is the largest k with
        (1-b2) * (1-qbar)^k  >  (1-b_star).
    A DIAGNOSTIC: reported next to the answer, never in the interval.
    """
    if b_star >= 1.0:
        return 10 ** 6          # certainty is unassailable; sentinel "infinite"
    ratio = (1 - b_star) / max(1 - b2, 1e-12)
    return max(0, math.ceil(math.log(ratio) / math.log(1 - qbar)) - 1)


def corroborate(mentions_by_value: dict[str, list],
                source_lookup: dict[str, Source],
                qtable: QTable | None = None) -> CorroboratedValue:
    """Adopt one value for one attribute of one record (guide 8.3).

    Args:
        mentions_by_value: competing CANONICAL values for ONE attribute,
            each mapped to the Mentions asserting it,
            e.g. {"$40M": [m1, m2], "$35M": [m3]}. Keys are assumed
            canonicalized upstream (pipeline runs canonicalize_value()
            before grouping).
        source_lookup: source_id -> Source for every referenced mention.
        qtable: reliability model; defaults to a fresh fixed-prior QTable.

    Returns:
        CorroboratedValue for the adopted value v*, carrying belief, nu
        (independent-origin count), component sizes, competing beliefs,
        version/supersession bookkeeping, and the margin kappa.
    """
    qtable = qtable or QTable()

    # (1) attribute-level derivation graph over ALL asserting sources
    deriv_G = build_attribute_graph(mentions_by_value, source_lookup)

    # (2) version partition: kill superseded versions and their echoes
    live, excluded, dead_srcs = live_values(
        mentions_by_value, source_lookup, deriv_G, qtable)
    if not live:
        # defensive: pathological input (e.g. every source somehow dead).
        # Fall back to treating everything as live rather than crashing;
        # supersession normally guarantees the newest chain doc stays live.
        live = set(mentions_by_value)
        dead_srcs = set()
        excluded = {v: 0 for v in mentions_by_value}

    # (3) noisy-OR belief per LIVE value over its components
    results, info, supporting = {}, {}, {}
    for value, mentions in mentions_by_value.items():
        if value not in live:
            continue                # stale: DISQUALIFIED, not outvoted
        alive_ms = [m for m in mentions if m.source_id not in dead_srcs]
        G = per_value_subgraph(deriv_G, alive_ms)
        comps = list(nx.weakly_connected_components(G))
        comp_q = []
        for comp in comps:          # ONE reliability per component (origin)
            anchored = any(source_lookup[sid].identity_anchored for sid in comp)
            qc = max(qtable.q(source_lookup[sid]) for sid in comp)
            # belt-and-braces: even if a capped q slipped through, an
            # unanchored COMPONENT as a whole never exceeds qbar. A copy
            # farm agreeing with itself is still one <=qbar witness.
            comp_q.append(qc if anchored else min(qc, qtable.qbar))
        belief = 1.0
        for q in comp_q:
            belief *= (1 - q)       # noisy-OR: all origins must be wrong
        results[value] = 1 - belief
        info[value] = {"nu": len(comps), "sizes": [len(c) for c in comps]}
        supporting[value] = [m.mention_id for m in alive_ms]

    # (4) adopt argmax belief; report the margin against the runner-up
    v_star = max(results, key=results.get)
    b_star = results[v_star]
    b2 = max([b for v, b in results.items() if v != v_star], default=0.0)
    return CorroboratedValue(
        value=v_star,
        belief=b_star,
        nu=info[v_star]["nu"],
        component_sizes=info[v_star]["sizes"],
        competing={v: b for v, b in results.items() if v != v_star},
        version_id=_version_of(v_star, mentions_by_value, source_lookup),
        n_dead_excluded=sum(excluded.values()),
        kappa=margin(b_star, b2, qtable.qbar),
        # Provenance discipline: exactly which (live) mentions asserted the
        # adopted value, so a wrong number can be walked back
        # CorroboratedValue -> Mentions -> Sources -> URL in seconds.
        supporting_mention_ids=supporting[v_star],
    )


def _version_of(value: str, mentions_by_value: dict[str, list],
                source_lookup: dict[str, Source]) -> int:
    """Version index of the adopted value along its authority chain.

    0 = no supersession machinery involved (no chain, or the chain's first
    doc). k = the adopted value is asserted by the (k+1)-th document of
    some authority chain, i.e. it survived k supersessions. Simple, and
    enough for the audit trail the guide asks version_id to carry.
    """
    best = 0
    for m in mentions_by_value.get(value, []):
        s = source_lookup[m.source_id]
        if not s.authority_chain_id:
            continue
        chain = sorted(
            (x for x in source_lookup.values()
             if x.authority_chain_id == s.authority_chain_id),
            key=lambda x: x.publish_time or x.fetch_time)
        ids = [x.source_id for x in chain]
        if s.source_id in ids:
            best = max(best, ids.index(s.source_id))
    return best
