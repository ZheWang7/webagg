"""SIGMOD guide ch. 9 smoke tests -- one test per edit, fully OFFLINE.

No torch, no network, no API keys: the embedder is stubbed with a
deterministic fake, the LLM adjudicator with a lambda, and the claims
engine's session with a no-op. Failures point at the responsible file:

  test_correlation_clustering_*   -> entity_resolution.correlation_clustering
  test_fragile_pairs_recorded     -> entity_resolution.resolve_entities
  test_domain_block_and_log       -> entity_resolution.candidate_pairs_logged
  test_rekey_*                    -> frontier.rekey_strata
  test_count_sensitivity_*        -> claims.ClaimsEngine.checksum / .rekey

Run: pytest tests/test_er_ch9.py -v
"""
from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import webagg.entity_resolution as er                      # noqa: E402
from webagg.frontier import (FrontierState, StratumState,  # noqa: E402
                             rekey_strata)
from webagg.claims import ClaimsEngine, CoverageView       # noqa: E402
from webagg.type_defs import Source, Mention               # noqa: E402

_NOW = datetime(2025, 1, 1)


# --- offline stubs ----------------------------------------------------------

class _FakeEncoder:
    """Deterministic 16-dim vector from the name's hash -- identical names
    embed identically, different names differently. Enough for blocking."""
    def encode(self, name: str) -> np.ndarray:
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(16)


@pytest.fixture(autouse=True)
def _stub_embedder():
    er._embed = _FakeEncoder()          # embedder() returns this, never torch
    yield
    er._embed = None


def _src(sid: str, domain: str = "example.com") -> Source:
    return Source(source_id=sid, url=f"https://{domain}/a", domain=domain,
                  fetch_time=_NOW, publish_time=None, title=None,
                  main_text="test", formulation_id="f0")


def _men(mid: str, sid: str, surface: str) -> Mention:
    return Mention(mention_id=mid, source_id=sid, entity_surface=surface,
                   record_kind="funding_round", attribute="amount",
                   value="$1M", passage="p", extracted_at=_NOW)


class _PairMatcher(er.Matcher):
    """Matcher whose theta is looked up by the PAIR OF SURFACES, so tests
    control every edge exactly. Features are bypassed on purpose -- these
    tests exercise the clustering/band logic, not the feature code."""
    def __init__(self, table: dict[frozenset, float], **kw):
        super().__init__(**kw)
        self.table = table

    def score_pair(self, m_a, m_b) -> float:
        return self.table[frozenset((m_a.entity_surface, m_b.entity_surface))]


@pytest.fixture
def _pairwise_scoring(monkeypatch):
    """Route resolve_entities through score_pair: features() returns the two
    mentions untouched and score() unpacks them."""
    monkeypatch.setattr(er, "features", lambda a, b, lookup: (a, b))
    monkeypatch.setattr(_PairMatcher, "score",
                        lambda self, x: self.score_pair(*x))


# --- 9.5  correlation clustering -------------------------------------------

def test_correlation_clustering_negative_edge_splits(_pairwise_scoring):
    """A confident split must survive: connected components would merge
    A-B and then chain anything touching them; correlation clustering lets
    the strong negative edges keep C out."""
    srcs = {"s1": _src("s1")}
    ms = [_men("mA", "s1", "Acme A"), _men("mB", "s1", "Acme B"),
          _men("mC", "s1", "Cinder")]
    matcher = _PairMatcher({frozenset(("Acme A", "Acme B")): 0.95,   # merge
                            frozenset(("Acme A", "Cinder")): 0.02,   # split
                            frozenset(("Acme B", "Cinder")): 0.02})  # split
    res = er.resolve_entities(ms, matcher, srcs,
                              adjudicator=lambda a, b, s: 0.5)
    e = res.mention_to_entity
    assert e["mA"] == e["mB"] and e["mC"] != e["mA"]
    assert res.fragile_pairs == []      # nothing sat in the band


def test_correlation_clustering_weighs_transitive_conflict(_pairwise_scoring):
    """Two strong merges (A-B, B-C) against one weaker split (A-C): the
    summed log-odds favor one cluster -- weights decide, not chains."""
    srcs = {"s1": _src("s1")}
    ms = [_men("mA", "s1", "X"), _men("mB", "s1", "Y"), _men("mC", "s1", "Z")]
    matcher = _PairMatcher({frozenset(("X", "Y")): 0.95,
                            frozenset(("Y", "Z")): 0.95,
                            frozenset(("X", "Z")): 0.10})
    e = er.resolve_entities(ms, matcher, srcs,
                            adjudicator=lambda a, b, s: 0.5).mention_to_entity
    # 2 * logit(0.95) ~ +5.9 beats logit(0.10) ~ -2.2: one entity
    assert len(set(e.values())) == 1


def test_fragile_pairs_recorded_whatever_the_adjudicator_says(_pairwise_scoring):
    """A band pair is FRAGILE even when the LLM confidently resolves it --
    the checksum's count-sensitivity check needs to know the cheap signals
    could have gone either way."""
    srcs = {"s1": _src("s1")}
    ms = [_men("mA", "s1", "Acme"), _men("mB", "s1", "Acme Holdings")]
    matcher = _PairMatcher({frozenset(("Acme", "Acme Holdings")): 0.50})  # band
    res = er.resolve_entities(ms, matcher, srcs,
                              adjudicator=lambda a, b, s: 0.99)  # LLM: merge!
    assert res.mention_to_entity["mA"] == res.mention_to_entity["mB"]
    assert [(a, b) for a, b, _ in res.fragile_pairs] == [("mA", "mB")]


# --- 9.2  blocking: shared-domain block + predicate log ---------------------

def test_domain_block_and_predicate_log():
    """Surfaces that share NO name signal must still pair via the new
    shared-domain block, and the blocking log must name the predicate."""
    srcs = {"s1": _src("s1", "acme.com"), "s2": _src("s2", "acme.com")}
    ms = [_men("m1", "s1", "Zebra Widgets"), _men("m2", "s2", "Quark Pty")]
    pairs, log = er.candidate_pairs_logged(ms, srcs)
    assert ("m1", "m2") in pairs
    assert "domain" in log[("m1", "m2")]          # the logged predicate


# --- ch. 9 duty #1: re-key strata -------------------------------------------

def _state_two_surfaces() -> FrontierState:
    """'acme' and 'acme holdings' each hold one record, seen by DIFFERENT single
    formulations -- two singletons pre-ER."""
    st = FrontierState()
    st.T = 2
    st.covered["acme|funding_round"] = {"f1"}
    st.covered["acme holdings|funding_round"] = {"f2"}
    st.record_stratum["acme|funding_round"] = "acme"
    st.record_stratum["acme holdings|funding_round"] = "acme holdings"
    st.strata["acme"] = StratumState(name="acme", V=1.0, claimed_count=None)
    st.strata["acme holdings"] = StratumState(name="acme holdings", V=2.0,
                                          claimed_count=3)
    return st


def test_rekey_merges_singletons_into_a_doubleton():
    """The guide's exact scenario: a record covered once under each surface
    key becomes a doubleton after the merge, and U_hat/Chao move."""
    st = _state_two_surfaces()
    pool_pre = {"acme", "acme holdings"}
    assert st.f(1, pool_pre) == 2 and st.f(2, pool_pre) == 0   # two singletons
    u_before = st.U_hat(pool_pre)

    ms = [_men("m1", "s1", "Acme"), _men("m2", "s2", "Acme Holdings")]
    info = rekey_strata(st, {"m1": "ent_00000", "m2": "ent_00000"}, ms)

    pool = {"ent_00000"}
    assert st.N(pool) == 1                        # two surface records -> one
    assert st.f(1, pool) == 0 and st.f(2, pool) == 1   # the doubleton
    assert st.U_hat(pool) < u_before              # unseen-mass estimate drops
    assert info["n_merges"] == 1
    assert info["surface_to_entity"] == {"acme": "ent_00000",
                                         "acme holdings": "ent_00000"}
    # StratumState merge semantics: V summed, strongest brake kept
    S = st.strata["ent_00000"]
    assert S.V == 3.0 and S.claimed_count == 3 and S.certified is None


def test_rekey_leaves_unmerged_strata_alone():
    st = _state_two_surfaces()
    ms = [_men("m1", "s1", "Acme"), _men("m2", "s2", "Acme Holdings")]
    rekey_strata(st, {"m1": "ent_00000", "m2": "ent_00001"}, ms)  # ER: split
    assert st.N({"ent_00000"}) == 1 and st.N({"ent_00001"}) == 1
    assert st.f(1, {"ent_00000", "ent_00001"}) == 2   # still two singletons


# --- ch. 9 duty #2: count-sensitivity check ---------------------------------

class _StubSession:
    def merge(self, row):                 # ingest() persists; tests don't
        pass


class _StubClaim:
    """Duck-typed COUNT claim: only the fields ingest()/the brake read."""
    def __init__(self, stratum_surface, source_id, n):
        self.stratum_surface, self.source_id = stratum_surface, source_id
        self.functional, self.value_num, self.scope = "COUNT", float(n), ""
    def to_row(self):
        return object()                   # _StubSession.merge ignores it


def _engine_with_count_claims(g: str, n: int) -> ClaimsEngine:
    ce = ClaimsEngine(_StubSession())
    ce.ingest(_StubClaim(g, "srcA", n))   # two distinct sources -> the
    ce.ingest(_StubClaim(g, "srcB", n))   # provisional brake arms
    return ce


def test_count_sensitivity_vetoes_and_queues():
    """|D_g| == n_g but a fragile pair exists: do NOT certify -- flag the
    conflict and queue the pair for verification (guide ch. 9, 'ten lines')."""
    ce = _engine_with_count_claims("acme", 3)
    st = ce.checksum("acme", CoverageView(
        n_records=3, fragile_pairs=(("mA", "mB", 0.5),)))
    assert st.conflict and not st.certified
    assert ce.verification_queue == [("acme", "mA", "mB", 0.5)]
    # idempotent: re-checking must not double-queue
    ce.checksum("acme", CoverageView(n_records=3,
                                     fragile_pairs=(("mA", "mB", 0.5),)))
    assert len(ce.verification_queue) == 1


def test_count_match_without_fragile_pairs_is_clean():
    ce = _engine_with_count_claims("acme", 3)
    st = ce.checksum("acme", CoverageView(n_records=3))
    assert not st.conflict and st.gap is None     # §11's certify point


def test_claims_engine_rekey_merges_witnesses():
    """One COUNT witness under each surface: pre-rekey neither stratum has
    the >= 2 sources the brake needs; after rekey() they back one entity."""
    ce = ClaimsEngine(_StubSession())
    ce.ingest(_StubClaim("Acme", "srcA", 3))        # -> stratum "acme"
    ce.ingest(_StubClaim("Acme Holdings", "srcB", 3))  # -> "acme holdings"
    assert ce.provisional_claimed_count("acme") is None
    ce.rekey({"acme": "ent_00000", "acme holdings": "ent_00000"})
    assert ce.provisional_claimed_count("ent_00000") == 3
