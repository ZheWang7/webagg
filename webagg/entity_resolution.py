"""Entity Resolution -- SIGMOD guide ch. 9 (design paper Sec. 6).

The web has no join key: the same subsidiary appears as "Acme Corp",
"Acme, Inc.", "ACME", or only via a CEO name. To aggregate "per entity"
we must INFER the join. Three pieces (guide ch. 9):

  1. blocking   -- coarse blocks (name prefix, token bigrams, embedding-LSH
                   bands, shared domain) keep comparison sub-quadratic; the
                   blocking miss rate delta_B <= (1 - p)^L is tuned via the
                   number of LSH bands L. Every candidate pair logs WHICH
                   predicate produced it ("log the predicate").
  2. matching   -- cheap features -> a CALIBRATED classifier
                   theta(x, y) = Pr[x == y]. Calibration is what turns the
                   per-pair error alpha into a number; alpha now feeds the
                   single fidelity certificate (Sec. 13) and every checksum's
                   "no false merge/split" condition. Band pairs
                   (tau_minus, tau_plus) escalate to an LLM adjudicator.
  3. clustering -- CORRELATION CLUSTERING over signed edge weights
                   log(theta / (1 - theta)) for transitive consistency.
                   Confident splits are NEGATIVE edges, so a chain of weak
                   merges cannot silently swallow a confident non-match.

Chapter-9 integration duties (consumed by frontier.rekey_strata and the
claims engine's count-sensitivity check):
  * fragile pairs: every in-band pair is remembered, whatever the
    adjudicator decided -- one flip there could change a stratum's COUNT.
  * mention_to_entity: the inferred join key {mention_id -> entity_id}.

Output object: ERResult (mention_to_entity, fragile_pairs, blocking_log,
alpha). `cluster_entities()` is kept as a thin back-compat wrapper that
returns just the dict, so pre-ch.9 callers and tests keep working.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import networkx as nx  # noqa: F401  (kept: some notebooks import via this module)
from rapidfuzz.fuzz import token_set_ratio, partial_ratio
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

# --- 9.1  embedding + normalization ----------------------------------------
_embed = None


def embedder():
    """Lazily build the sentence-transformer. Import is deferred to first use so
    `import webagg.entity_resolution` does NOT pull in torch, and tests can stub
    `entity_resolution._embed` with a fake encoder."""
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer  # lazy: torch
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed


def normalize_name(s: str) -> str:
    """Canonical, order-independent token form of an entity surface.

    NOTE (repo deviation, carried over from the pre-SIGMOD module): the
    guide's literal snippet stringifies a Python list; we implement its
    stated intent ("tokenized sorted form"): strip the corporate suffix,
    drop punctuation, join SORTED tokens with a space.
    """
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\b(inc|corp|corporation|llc|ltd|limited|co|company)\b\.?", "", s)
    toks = re.sub(r"[^\w\s]", " ", s).split()
    return " ".join(sorted(toks))


# --- 9.2  blocking ----------------------------------------------------------

LSH_BANDS = 2   # L in the guide's bound: a true pair is missed w.p.
                # delta_B <= (1 - p)^L, p = per-band collision probability.
                # Raise L to shrink delta_B (more candidate pairs, more cost).


def lsh_miss_bound(p_band: float, L: int = LSH_BANDS) -> float:
    """delta_B <= (1 - p)^L: probability that L independent LSH bands ALL
    fail to co-bucket a true pair (guide ch. 9, blocking bullet). A tuning
    diagnostic, not a certificate -- log it so the choice of L is auditable."""
    return (1.0 - p_band) ** L


def lsh_bucket(vec: np.ndarray, band: int, n_bits: int = 8) -> str:
    """Random-hyperplane LSH: sign bits of n_bits planes -> an n_bits-char key.
    Seeding by `band` makes each band's planes deterministic across runs."""
    rng = np.random.default_rng(seed=band)
    planes = rng.standard_normal((n_bits, len(vec)))
    bits = (planes @ vec > 0).astype(int)
    return "".join(map(str, bits))


def blocks_for(mention, source_lookup: dict | None = None) -> list[str]:
    """Return the coarse blocks this mention belongs to. Two mentions are
    compared only if they share at least one block. Block keys are
    self-describing ("prefix:", "bigram:", "lsh:", "domain:") -- that prefix
    IS the logged predicate the guide asks for.

    NEW in ch. 9: the shared-domain block ("assign each mention to coarse
    blocks (normalized-name prefix, embedding-LSH bucket, shared domain)").
    Needs `source_lookup` (source_id -> Source) to read the domain; omitted
    when the lookup is not supplied (back-compat with pre-ch.9 callers).
    """
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
    # 3. LSH blocks on the embedding (L = LSH_BANDS bands)
    vec = embedder().encode(name)
    for band in range(LSH_BANDS):
        blocks.append(f"lsh:{band}:{lsh_bucket(vec, band=band)}")
    # 4. shared-domain block (ch. 9): two mentions from the same site are
    # worth comparing even when the surfaces look nothing alike ("ACME"
    # in the header vs. "Acme, Inc." in the footer of one press page).
    if source_lookup is not None:
        src = source_lookup.get(mention.source_id)
        if src is not None and src.domain:
            blocks.append(f"domain:{src.domain}")
    return blocks


def candidate_pairs_logged(mentions: list, source_lookup: dict | None = None
                           ) -> tuple[set[tuple[str, str]],
                                      dict[tuple[str, str], set[str]]]:
    """All within-block (mention_id, mention_id) pairs, PLUS the blocking log:
    {pair -> set of predicates ('prefix', 'bigram', 'lsh', 'domain') that
    proposed it}. The log is what makes delta_B auditable -- you can see which
    predicate is doing the work and which is dead weight."""
    buckets: dict[str, list[str]] = {}
    for m in mentions:
        for b in blocks_for(m, source_lookup):
            buckets.setdefault(b, []).append(m.mention_id)
    cand: set[tuple[str, str]] = set()
    log: dict[tuple[str, str], set[str]] = {}
    for bkey, ids in buckets.items():
        predicate = bkey.split(":", 1)[0]          # "prefix" / "bigram" / ...
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                if a == b:
                    continue
                cand.add((a, b))
                log.setdefault((a, b), set()).add(predicate)
    return cand, log


def candidate_pairs(mentions: list, source_lookup: dict | None = None
                    ) -> set[tuple[str, str]]:
    """Back-compat shim (pre-ch.9 signature): pairs only, no log."""
    return candidate_pairs_logged(mentions, source_lookup)[0]


# --- 9.3  pairwise matching -------------------------------------------------

def temporal_plausibility(m_a, m_b, source_lookup: dict) -> float:
    """Ch.-9 feature: are the two assertions chronologically compatible?

    FLAGGED REPO CHOICE: the guide names "temporal plausibility" as a
    feature but does not define it. We use the valid time t_asof (falling
    back to the source's publish_time), and score exp(-|delta_days| / 1825):
    1.0 for same-day facts, ~0.37 at five years apart, decaying after. When
    either side has no timestamp we return the NEUTRAL value 0.5 so a
    missing date neither helps nor hurts the match."""
    def _t(m) -> datetime | None:
        if m.t_asof is not None:
            return m.t_asof
        src = source_lookup.get(m.source_id)
        return getattr(src, "publish_time", None) if src else None

    ta, tb = _t(m_a), _t(m_b)
    if ta is None or tb is None:
        return 0.5
    days = abs((ta - tb).total_seconds()) / 86400.0
    return math.exp(-days / 1825.0)          # 5-year decay scale


def features(m_a, m_b, source_lookup) -> np.ndarray:
    """Cheap pairwise signals feeding theta(x, y) (guide ch. 9 matching
    bullet): name similarity, partial similarity, shared domain, embedding
    cosine, temporal plausibility. `source_lookup` maps source_id -> Source.

    (The guide also lists shared CEO/address; our extraction schema does not
    carry those attributes yet, so shared-domain stands in for the "shared
    side-channel identifier" signal -- flagged deviation.)"""
    name_sim = token_set_ratio(m_a.entity_surface, m_b.entity_surface) / 100.0
    part_sim = partial_ratio(m_a.entity_surface, m_b.entity_surface) / 100.0
    same_domain = float(source_lookup[m_a.source_id].domain ==
                        source_lookup[m_b.source_id].domain)
    emb_a = embedder().encode(m_a.entity_surface)
    emb_b = embedder().encode(m_b.entity_surface)
    emb_cos = float(np.dot(emb_a, emb_b) /
                    (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
    temp = temporal_plausibility(m_a, m_b, source_lookup)
    return np.array([name_sim, part_sim, same_domain, emb_cos, temp])


class Matcher:
    """theta(x, y) = P[x == y | features].

    Cold-start uses a hand-tuned linear combination. Once you have the
    ~200-pair hand-labeled set the guide asks for, call .fit(X, y):
    it wraps logistic regression in CalibratedClassifierCV (sigmoid /
    Platt scaling) so theta BEHAVES LIKE A PROBABILITY -- that calibration
    is what makes the per-pair error alpha a number, and alpha now appears
    in the fidelity interval (Sec. 13) and in every checksum's conditions.

    Thresholds (guide ch. 9 pseudocode defaults): theta >= tau_plus merge,
    theta <= tau_minus split, and the band in between escalates to the LLM
    adjudicator -- the expensive call only where cheap signals fail.
    NOTE: tau_minus moves 0.20 -> 0.15 to match the SIGMOD pseudocode.
    """

    def __init__(self, tau_minus: float = 0.15, tau_plus: float = 0.85):
        self.tau_minus, self.tau_plus = tau_minus, tau_plus
        self.clf = None          # CalibratedClassifierCV | None, set by .fit()
        self.alpha: float | None = None   # cross-validated per-pair error rate

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit + calibrate on the hand-labeled pair set, and estimate alpha
        (the matcher's per-pair error) by cross-validation on the SAME set --
        an honest out-of-fold estimate, not training accuracy."""
        y = np.asarray(y)
        # cv folds cannot exceed the size of the rarer class
        n_min_class = int(min(np.bincount(y.astype(int))))
        cv = max(2, min(3, n_min_class))
        base = LogisticRegression(max_iter=1000)
        self.clf = CalibratedClassifierCV(base, method="sigmoid", cv=cv).fit(X, y)
        oof = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv)
        self.alpha = float(np.mean(oof != y))
        return self

    def score(self, x: np.ndarray) -> float:
        if self.clf is None:
            # cold start: name_sim, part_sim, same_domain, emb_cos, temporal
            return float(0.40 * x[0] + 0.15 * x[1] + 0.10 * x[2]
                         + 0.25 * x[3] + 0.10 * x[4])
        return float(self.clf.predict_proba(x.reshape(1, -1))[0, 1])


# --- 9.4  LLM adjudicator for the escalation band --------------------------

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
    out = call_llm(system=sys, user=user, purpose="er_adjudication")["payload"]
    return float(out["confidence"]) if out["match"] else 1 - float(out["confidence"])


# --- 9.5  correlation clustering -------------------------------------------

_P_CLIP = 1e-3   # keep log-odds finite: p in [_P_CLIP, 1 - _P_CLIP]


def weight(p: float) -> float:
    """Signed log-odds edge weight log(theta / (1 - theta)) (guide ch. 9
    clustering bullet). Positive when theta > 0.5 (evidence FOR same
    entity), negative below. Clipped so a 0/1 score cannot produce inf."""
    p = min(max(p, _P_CLIP), 1.0 - _P_CLIP)
    return math.log(p / (1.0 - p))


def correlation_clustering(node_ids: list[str],
                           edges: dict[tuple[str, str], float]) -> dict[str, str]:
    """Greedy correlation clustering over the SIGNED edge graph.

    Why not connected components (the pre-ch.9 behavior): components ignore
    negative edges, so weak merges A-B and B-C silently swallow a CONFIDENT
    split A-C. Correlation clustering weighs agreement: a cluster is good
    when the total signed weight inside it is positive.

    Two phases, both deterministic (sorted iteration order):
      1. PIVOT: walk nodes in sorted order; an unassigned node opens a
         cluster and absorbs every unassigned neighbor joined by a
         positive-weight edge (Ailon-style pivot on the '+' graph).
      2. LOCAL MOVES: a few sweeps where each node moves to the cluster
         (or a fresh singleton) maximizing its summed signed weight to the
         members -- this is where negative edges get to VETO bad merges
         that phase 1's greed produced.
    Missing edges count as weight 0 (blocking already declared those pairs
    not worth comparing).
    """
    # symmetric adjacency for O(1) neighbor lookups
    adj: dict[str, dict[str, float]] = {u: {} for u in node_ids}
    for (a, b), w in edges.items():
        adj[a][b] = w
        adj[b][a] = w

    # --- phase 1: greedy pivot on positive edges ---------------------------
    cluster_of: dict[str, int] = {}
    next_cid = 0
    for u in sorted(node_ids):
        if u in cluster_of:
            continue
        cid = next_cid
        next_cid += 1
        cluster_of[u] = cid
        for v, w in sorted(adj[u].items()):
            if v not in cluster_of and w > 0:
                cluster_of[v] = cid

    # --- phase 2: local-move refinement (negative edges veto) --------------
    for _sweep in range(3):                       # few sweeps suffice at our scale
        moved = False
        for u in sorted(node_ids):
            # summed signed weight from u into each neighboring cluster
            gain: dict[int, float] = {}
            for v, w in adj[u].items():
                gain[cluster_of[v]] = gain.get(cluster_of[v], 0.0) + w
            own = cluster_of[u]
            # u's own cluster score must not count self (no self-edges exist)
            best_cid, best_val = own, gain.get(own, 0.0)
            for cid, val in sorted(gain.items()):
                if val > best_val:
                    best_cid, best_val = cid, val
            if best_val < 0:                      # everyone repels u
                best_cid = next_cid               # -> fresh singleton
            if best_cid != own:
                if best_cid == next_cid:
                    next_cid += 1
                cluster_of[u] = best_cid
                moved = True
        if not moved:
            break

    # dense, deterministic entity ids: ent_00000, ent_00001, ...
    out: dict[str, str] = {}
    relabel: dict[int, str] = {}
    for u in sorted(node_ids):
        cid = cluster_of[u]
        if cid not in relabel:
            relabel[cid] = f"ent_{len(relabel):05d}"
        out[u] = relabel[cid]
    return out


# --- 9.6  the full ER pass --------------------------------------------------

@dataclass
class ERResult:
    """Everything downstream layers need from one ER pass (ch. 9).

    mention_to_entity : the inferred join key {mention_id -> entity_id}
    fragile_pairs     : [(mid_a, mid_b, theta)] whose decision sat in the
                        band (tau_minus, tau_plus) -- REGARDLESS of how the
                        adjudicator ruled. One flip there can change a
                        stratum's COUNT, so the checksum's count-sensitivity
                        check (claims.py) must see them.
    blocking_log      : {pair -> predicates that proposed it} ("log the
                        predicate", ch. 9 blocking bullet)
    alpha             : the matcher's calibrated per-pair error (None on
                        cold start) -- feeds the Sec.-13 fidelity interval.
    """
    mention_to_entity: dict[str, str]
    fragile_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    blocking_log: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    alpha: float | None = None

    def fragile_by_stratum(self) -> dict[str, list[tuple[str, str, float]]]:
        """Group fragile pairs by post-ER stratum (= entity_id). A pair is
        charged to EVERY stratum containing one of its endpoints: whichever
        way the flip goes, that stratum's count is the one at risk."""
        out: dict[str, list[tuple[str, str, float]]] = {}
        for (a, b, th) in self.fragile_pairs:
            for mid in (a, b):
                g = self.mention_to_entity.get(mid)
                if g is not None:
                    out.setdefault(g, [])
                    if (a, b, th) not in out[g]:
                        out[g].append((a, b, th))
        return out


def resolve_entities(mentions, matcher: Matcher, source_lookup,
                     adjudicator=adjudicate_llm) -> ERResult:
    """The ch.-9 ER pass: block -> match -> adjudicate the band -> correlation
    clustering. Returns an ERResult (see above).

    Signed-edge construction mirrors the guide's pseudocode:
        theta >= tau_plus            ->  +weight(theta)   confident merge
        theta <= tau_minus           ->  -weight is ALREADY negative
                                         (log-odds of a small theta)
        tau_minus < theta < tau_plus ->  adjudicate; edge = weight(theta_LLM);
                                         the pair is recorded as FRAGILE.

    `adjudicator` is injectable (defaults to the real adjudicate_llm) so
    unit tests can pass a deterministic stub -- no network, no torch.
    """
    pairs, blocking_log = candidate_pairs_logged(mentions, source_lookup)
    by_id = {m.mention_id: m for m in mentions}
    edges: dict[tuple[str, str], float] = {}
    fragile: list[tuple[str, str, float]] = []
    for (a, b) in sorted(pairs):
        x = features(by_id[a], by_id[b], source_lookup)
        theta = matcher.score(x)
        if theta >= matcher.tau_plus:
            edges[(a, b)] = weight(theta)             # strong +
        elif theta <= matcher.tau_minus:
            edges[(a, b)] = weight(theta)             # strong - (log-odds < 0)
        else:
            # the band: cheap signals failed -> spend the LLM call, and
            # remember the pair as fragile whatever the verdict is.
            theta2 = adjudicator(by_id[a], by_id[b], source_lookup)
            edges[(a, b)] = weight(theta2)
            fragile.append((a, b, theta))
    m2e = correlation_clustering(sorted(by_id.keys()), edges)
    return ERResult(mention_to_entity=m2e, fragile_pairs=fragile,
                    blocking_log=blocking_log, alpha=matcher.alpha)


def cluster_entities(mentions, matcher: Matcher, source_lookup,
                     adjudicator=adjudicate_llm) -> dict[str, str]:
    """Back-compat wrapper (pre-ch.9 signature): {mention_id: entity_id} only.
    New callers should use resolve_entities() and keep the ERResult -- the
    fragile pairs and the blocking log are ch.-9 integration duties."""
    return resolve_entities(mentions, matcher, source_lookup,
                            adjudicator=adjudicator).mention_to_entity
