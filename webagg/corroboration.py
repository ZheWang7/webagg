"""
Section 7 -- The Corroboration Layer.

For one attribute of one resolved record, we are given the competing VALUES
asserted for it, and for each value the set of Mentions (with their source) that
assert it. We build a provenance graph G_v over those sources, find connected
components (= independent witnesses), and compute a noisy-OR belief so that
N echoing copies of one press release count as ONE witness, not N.
"""
import re
import networkx as nx
from rapidfuzz.fuzz import token_set_ratio
from datasketch import MinHash, MinHashLSH
from webagg.type_defs import Source, CorroboratedValue

SHINGLE_K = 5  # 5-grams of tokens


# --- 7.2  derivation-edge detection ----------------------------------------
def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def shingle_jaccard(a: str, b: str) -> float:
    A, B = shingles(a), shingles(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def longest_common_verbatim_tokens(a: str, b: str) -> int:
    """Length of the longest matching contiguous token run."""
    A = re.findall(r"\w+", a.lower())
    B = re.findall(r"\w+", b.lower())
    best = 0
    dp = [0] * (len(B) + 1)
    for i, ta in enumerate(A, 1):
        new = [0] * (len(B) + 1)
        for j, tb in enumerate(B, 1):
            if ta == tb:
                new[j] = dp[j - 1] + 1
                best = max(best, new[j])
        dp = new
    return best


def derivation_edge(s_i: Source, s_j: Source,
                    passage_i: str, passage_j: str,
                    sim_threshold: float = 0.85) -> bool:
    # 1. temporal: s_j must be strictly later than s_i to be derived FROM it
    ti = s_i.publish_time or s_i.fetch_time
    tj = s_j.publish_time or s_j.fetch_time
    if tj <= ti:
        return False
    # 2. near-duplicate passage
    if shingle_jaccard(passage_i, passage_j) >= sim_threshold:
        return True
    # 3. explicit attribution (str() because Source.url is an HttpUrl object)
    if str(s_i.url) in (s_j.main_text or "") or s_i.domain in (s_j.main_text or ""):
        return True
    # 4. long verbatim run
    if longest_common_verbatim_tokens(passage_i, passage_j) >= 25:
        return True
    return False


# --- 7.3  provenance graph + noisy-OR --------------------------------------
def reliability(source: Source) -> float:
    """Prior reliability q(c) per the design document. Tune empirically."""
    DOMAIN_PRIORS = {
        "sec.gov": 0.97, "clinicaltrials.gov": 0.97, "uspto.gov": 0.95,
        "reuters.com": 0.80, "bloomberg.com": 0.80,
        "techcrunch.com": 0.60, "crunchbase.com": 0.65,
    }
    return DOMAIN_PRIORS.get(source.domain, 0.50)


def build_graph(mentions_for_value, source_lookup):
    """mentions_for_value: list[Mention] all asserting the same value."""
    G = nx.DiGraph()
    for m in mentions_for_value:
        G.add_node(m.source_id,
                   source=source_lookup[m.source_id], passage=m.passage)
    nodes = list(G.nodes(data=True))
    for i, (id_i, d_i) in enumerate(nodes):
        for id_j, d_j in nodes[i + 1:]:
            if derivation_edge(d_i["source"], d_j["source"],
                               d_i["passage"], d_j["passage"]):
                G.add_edge(id_i, id_j)
            elif derivation_edge(d_j["source"], d_i["source"],
                                 d_j["passage"], d_i["passage"]):
                G.add_edge(id_j, id_i)
    return G


def corroborate(mentions_by_value: dict[str, list],
                source_lookup: dict[str, Source]) -> CorroboratedValue:
    """
    Decide the adopted value across competing candidates.

    Args:
    - mentions_by_value: Competing values for ONE attribute, each mapped to
            the list of Mentions asserting it,
            e.g. {"$40M": [m1, m2], "$50M": [m3]}. Buckets are assumed pre-grouped
            (one per distinct candidate value) by the upstream resolution step.
    - source_lookup: Maps each Mention's source_id to its full Source, so the
            graph can read domains, timestamps, URLs, and text. Must contain an
            entry for every source_id referenced in mentions_by_value.

    Return:
    - A CorroboratedValue for the adopted value, carrying its belief, nu (the
        number of independent witnesses), component_sizes (sources per witness),
        and competing (each rejected value mapped to its own belief).
    """
    results = {}
    component_info = {}
    for value, mentions in mentions_by_value.items():
        G = build_graph(mentions, source_lookup)
        components = list(nx.weakly_connected_components(G))
        # noisy-OR: each component is one independent witness; use its
        # highest-reliability source as that witness's q.
        comp_q = []
        for comp in components:
            qs = [reliability(source_lookup[sid]) for sid in comp]
            comp_q.append(max(qs))
        belief = 1.0
        for q in comp_q:
            belief *= (1 - q)
        belief = 1 - belief
        results[value] = belief
        component_info[value] = {"nu": len(components),
                                 "comp_sizes": [len(c) for c in components]}
    v_star = max(results, key=results.get)
    return CorroboratedValue(
        value=v_star,
        belief=results[v_star],
        nu=component_info[v_star]["nu"],
        component_sizes=component_info[v_star]["comp_sizes"],
        competing={v: b for v, b in results.items() if v != v_star},
        # Provenance discipline: record exactly which mentions
        # asserted the adopted value, so a wrong number can be walked back
        # CorroboratedValue -> Mentions -> Sources -> URL in seconds.
        supporting_mention_ids=[m.mention_id for m in mentions_by_value[v_star]],
    )
