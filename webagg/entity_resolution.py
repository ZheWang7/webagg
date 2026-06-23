"""
The web has no join key: the same subsidiary appears as "Acme Corp", "Acme, Inc.",
"ACME", or only via a CEO name. To aggregate "per entity" we must INFER the join.
Three pieces:
  1. blocking: keep pairwise comparison sub-quadratic
  2. matching: a scored pairwise classifier with an LLM-adjudicated band
  3. clustering: correlation clustering for transitive consistency
Output: a stable {mention_id -> entity_id} map - the inferred join key.
"""
from __future__ import annotations
import re
import unicodedata
from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
from rapidfuzz.fuzz import token_set_ratio, partial_ratio
from sklearn.linear_model import LogisticRegression

# --- 8.2  embedding + normalization ----------------------------------------
_embed = None


def embedder():
    """Lazily build the sentence-transformer. Import is deferred to first use so
    `import webagg.entity_resolution` does NOT pull in torch, and tests can stub
    `entity_resolution._embed` with a fake encoder."""
    global _embed
    if _embed is None:
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed


def normalize_name(s: str) -> str:
    """Canonical, order-independent token form of an entity surface.

    NOTE (repo deviation): the guide's literal last line is
        re.sub(r"[^\\w\\s]", " ", s).split().__str__()
    which stringifies a Python list ("['acme', 'corp']") and is neither sorted
    nor clean -- it poisons the prefix/bigram blocks downstream. The comment in
    the guide says "tokenized sorted form", so we implement that intent: strip
    the corporate suffix, drop punctuation, and join SORTED tokens with a space.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\b(inc|corp|corporation|llc|ltd|limited|co|company)\b\.?", "", s)
    toks = re.sub(r"[^\w\s]", " ", s).split()
    return " ".join(sorted(toks))


def lsh_bucket(vec: np.ndarray, band: int, n_bits: int = 8) -> str:
    """Random-hyperplane LSH: sign bits of n_bits planes -> an n_bits-char key.
    Seeding by `band` makes each band's planes deterministic across runs."""
    rng = np.random.default_rng(seed=band)
    planes = rng.standard_normal((n_bits, len(vec)))
    bits = (planes @ vec > 0).astype(int)
    return "".join(map(str, bits))


def blocks_for(mention) -> list[str]:
    """Return the coarse blocks this mention belongs to. Two mentions are
    compared only if they share at least one block."""
    name = mention.entity_surface
    norm = normalize_name(name)
    blocks: list[str] = []
    # 1. prefix block (first 4 chars of the de-spaced normalized name)
    n = re.sub(r"\s", "", norm)
    if len(n) >= 4:
        blocks.append(f"prefix:{n[:4]}")
    # 2. token bigrams
    toks = norm.split()
    for i in range(len(toks) - 1):
        blocks.append(f"bigram:{toks[i]}_{toks[i + 1]}")
    # 3. LSH blocks on the embedding (2 bands suffice for the prototype)
    vec = embedder().encode(name)
    blocks.append(f"lsh:{lsh_bucket(vec, band=0)}")
    blocks.append(f"lsh:{lsh_bucket(vec, band=1)}")
    return blocks


def candidate_pairs(mentions: list) -> set[tuple[str, str]]:
    """All within-block (mention_id, mention_id) pairs, deduplicated/ordered."""
    buckets: dict[str, list[str]] = {}
    for m in mentions:
        for b in blocks_for(m):
            buckets.setdefault(b, []).append(m.mention_id)
    cand: set[tuple[str, str]] = set()
    for ids in buckets.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                if a != b:
                    cand.add((a, b))
    return cand


# --- 8.3  pairwise matching -------------------------------------------------
def features(m_a, m_b, source_lookup) -> np.ndarray:
    """Cheap pairwise signals: name similarity, partial similarity, shared
    domain, embedding cosine. `source_lookup` maps source_id -> Source."""
    name_sim = token_set_ratio(m_a.entity_surface, m_b.entity_surface) / 100.0
    part_sim = partial_ratio(m_a.entity_surface, m_b.entity_surface) / 100.0
    same_domain = float(source_lookup[m_a.source_id].domain ==
                        source_lookup[m_b.source_id].domain)
    emb_a = embedder().encode(m_a.entity_surface)
    emb_b = embedder().encode(m_b.entity_surface)
    emb_cos = float(np.dot(emb_a, emb_b) /
                    (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
    return np.array([name_sim, part_sim, same_domain, emb_cos])


class Matcher:
    """theta(x, y) = P[x == y | features]. Cold-start uses a hand-tuned linear
    combination; once you label pairs, call .fit() to swap in logistic
    regression (calibrate it so theta behaves like a probability)."""

    def __init__(self, tau_minus: float = 0.20, tau_plus: float = 0.85):
        self.tau_minus, self.tau_plus = tau_minus, tau_plus
        self.clf = None  # LogisticRegression | None, set by .fit()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf = LogisticRegression(max_iter=1000).fit(X, y)

    def score(self, x: np.ndarray) -> float:
        if self.clf is None:
            # name_sim, part_sim, same_domain, emb_cos
            return float(0.45 * x[0] + 0.20 * x[1] + 0.10 * x[2] + 0.25 * x[3])
        return float(self.clf.predict_proba(x.reshape(1, -1))[0, 1])


# --- 8.4  LLM adjudicator for the escalation band --------------------------
def adjudicate_llm(m_a, m_b, source_lookup) -> float:
    """Spend an LLM call only on band pairs. Returns a probability in [0,1]:
    confidence if match, else 1 - confidence."""
    from .llm import call_llm  # lazy: keeps API-key/client construction out of import
    sys = open("prompts/match_adjudicator.txt").read()
    s_a = source_lookup[m_a.source_id]
    s_b = source_lookup[m_b.source_id]
    user = ("A:\n"
            f"surface: {m_a.entity_surface}\ndomain: {s_a.domain}\n"
            f"context: {m_a.passage}\n\n"
            "B:\n"
            f"surface: {m_b.entity_surface}\ndomain: {s_b.domain}\n"
            f"context: {m_b.passage}")
    out = call_llm(system=sys, user=user)["payload"]
    return float(out["confidence"]) if out["match"] else 1 - float(out["confidence"])


# --- 8.5  clustering --------------------------------------------------------
def cluster_entities(mentions, matcher: Matcher, source_lookup,
                     adjudicator=adjudicate_llm) -> dict[str, str]:
    """Resolve mentions into entities. Returns {mention_id: entity_id}.

    Confident merges (theta >= tau_plus) and band pairs the adjudicator confirms
    (>= 0.5) become edges; connected components of that merge graph are entities.

    NOTE (repo deviation): `adjudicator` is injectable (defaults to the real
    `adjudicate_llm`) so unit tests can pass a deterministic stub and avoid any
    network/LLM call.
    """
    pairs = candidate_pairs(mentions)
    by_id = {m.mention_id: m for m in mentions}
    G = nx.Graph()
    G.add_nodes_from(by_id.keys())
    for (a, b) in pairs:
        x = features(by_id[a], by_id[b], source_lookup)
        theta = matcher.score(x)
        if theta >= matcher.tau_plus:
            G.add_edge(a, b, theta=theta, kind="auto")
        elif matcher.tau_minus < theta < matcher.tau_plus:
            theta2 = adjudicator(by_id[a], by_id[b], source_lookup)
            if theta2 >= 0.5:
                G.add_edge(a, b, theta=theta2, kind="adjudicated")
        # theta <= tau_minus -> confident split, no edge
    out: dict[str, str] = {}
    for i, comp in enumerate(nx.connected_components(G)):
        eid = f"ent_{i:05d}"
        for mid in comp:
            out[mid] = eid
    return out
