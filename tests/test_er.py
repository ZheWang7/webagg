"""
Section 8 -- INTEGRATION tests for the entity-resolution layer.

Unlike test_entity_resolution.py (which stubs the embedder and the LLM so it runs
offline in milliseconds), this file exercises the REAL components:

  * the real `all-MiniLM-L6-v2` sentence-transformer (first run downloads ~90 MB
    from Hugging Face), and
  * real `gpt-5-nano` calls through your own `call_llm` (costs a few cents and a
    few seconds, and needs API keys).

Run all of it from the PROJECT ROOT (so the relative prompt path
`prompts/match_adjudicator.txt` and your `.env` resolve):

    pytest tests/test_entity_resolution_integration.py -v -s

Selective running:
    pytest tests/test_entity_resolution_integration.py -k "real_embed" -v   # no API key, no cost
    pytest tests/test_entity_resolution_integration.py -k "not real_llm" -v # skip the paid LLM tests

Skips you may see, and what they mean:
  * "sentence-transformers not installed"  -> pip install sentence-transformers
  * "API keys not set"                     -> your `.env` needs OPENAI_API_KEY *and*
                                              ANTHROPIC_API_KEY. (Your llm.py builds
                                              BOTH clients at import, so both keys must
                                              exist even though ER only uses OpenAI.)
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from datetime import datetime

# Repo root = parent of tests/. Load .env from there BEFORE evaluating the skip
# guards, so the API-key checks see keys defined in the file.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from webagg.type_defs import Source, Mention   # noqa: E402
import webagg.entity_resolution as er          # noqa: E402

# --- skip guards ------------------------------------------------------------
_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
_HAS_KEYS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("ANTHROPIC_API_KEY"))

needs_embedder = pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
needs_llm = pytest.mark.skipif(
    not (_HAS_ST and _HAS_KEYS),
    reason="API keys not set (need OPENAI_API_KEY and ANTHROPIC_API_KEY) and/or "
           "sentence-transformers missing",
)


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """adjudicate_llm() does open('prompts/match_adjudicator.txt') relative to the
    CWD, so make sure every test runs as if launched from the project root."""
    monkeypatch.chdir(ROOT)


# --- Example 5 fixtures (real signals in the passages so the LLM can decide) -
_NOW = datetime(2025, 1, 1)   # fixed timestamp; ER ignores it, but the models require it
def _src(sid: str, domain: str) -> Source:
    return Source(
        source_id=sid,
        url=f"https://{domain}/article",
        domain=domain,
        fetch_time=_NOW,
        publish_time=None,
        title=None,
        main_text="(integration test source)",
        formulation_id="er-int-test",
    )


def _men(mid: str, sid: str, surface: str, passage: str) -> Mention:
    return Mention(
        mention_id=mid,
        source_id=sid,
        entity_surface=surface,
        record_kind="company",
        attribute="name",
        value=surface,
        passage=passage,
        extracted_at=_NOW,
    )


SRCS = {
    "s1": _src("s1", "techcrunch.com"),
    "s2": _src("s2", "reuters.com"),
    "s3": _src("s3", "acme.io"),
    "s4": _src("s4", "logistics-news.example"),
}
ACME = [
    _men("x1", "s1", "Acme Corp",
         "Acme Corp, the San Francisco software maker led by CEO Jane Lee, raised a round."),
    _men("x2", "s2", "Acme, Inc.",
         "Acme, Inc. of San Francisco, whose chief executive is Jane Lee, announced funding."),
    _men("x3", "s3", "ACME",
         "ACME's official site acme.io; the company is run by Jane Lee out of San Francisco."),
    _men("x4", "s4", "Acme Logistics",
         "Acme Logistics, a New York freight firm headed by CEO Raj Patel, expanded its fleet."),
]


# ===========================================================================
# Tier 1 -- real embedder only (no API key, no LLM cost)
# ===========================================================================
@needs_embedder
def test_real_embedder_shape_and_determinism():
    """The real model returns a fixed-width vector and is deterministic."""
    v1 = er.embedder().encode("Acme Corp")
    v2 = er.embedder().encode("Acme Corp")
    assert v1.shape == (384,)                     # all-MiniLM-L6-v2 dimension
    assert float((v1 == v2).all()) == 1.0          # deterministic for identical input


@needs_embedder
def test_real_embedding_geometry():
    """Name variants of the same company sit closer than a different company."""
    def cos(a, b):
        ea, eb = er.embedder().encode(a), er.embedder().encode(b)
        import numpy as np
        return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9))

    same = cos("Acme Corp", "Acme, Inc.")
    diff = cos("Acme Corp", "Zenith Robotics")
    print(f"\ncos(Acme Corp, Acme Inc)={same:.3f}  cos(Acme Corp, Zenith)={diff:.3f}")
    assert same > diff                             # co-referent names are nearer


@needs_embedder
def test_blocking_with_real_embeddings_coblocks_family():
    """All four acme* mentions share at least one block (the prefix block)."""
    cand = er.candidate_pairs(ACME)
    assert ("x1", "x2") in cand
    assert ("x1", "x3") in cand
    assert ("x1", "x4") in cand                    # co-blocked even though they differ


@needs_embedder
def test_matcher_coldstart_with_real_embeddings():
    """Cold-start scoring: identical names auto-merge; an unrelated company does not."""
    m = er.Matcher()
    identical = m.score(er.features(
        _men("a", "s1", "Acme Corp", ""), _men("b", "s1", "Acme Corp", ""), SRCS))
    unrelated = m.score(er.features(
        _men("a", "s1", "Acme Corp", ""),
        _men("b", "s4", "Zenith Robotics", ""), SRCS))
    print(f"\ncold-start identical={identical:.3f}  unrelated={unrelated:.3f}")
    assert identical >= m.tau_plus                 # >= 0.85 -> confident merge
    assert unrelated < 0.5                         # clearly not a merge


# ===========================================================================
# Tier 2 -- real LLM adjudicator (needs API keys; makes live gpt-5-nano calls)
# ===========================================================================
@needs_llm
def test_real_adjudicator_returns_probability():
    """adjudicate_llm returns a float in [0,1] using a live model call."""
    p = er.adjudicate_llm(ACME[0], ACME[1], SRCS)   # Acme Corp vs Acme, Inc.
    print(f"\nadjudicate(Acme Corp, Acme Inc) = {p:.3f}")
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


@needs_llm
def test_real_adjudicator_confirms_obvious_match():
    """Same CEO, same city, same company -> the LLM should call it a match (>=0.5)."""
    p = er.adjudicate_llm(ACME[0], ACME[1], SRCS)
    assert p >= 0.5


@needs_llm
def test_real_adjudicator_rejects_obvious_nonmatch():
    """Different CEO, different city, different domain -> not a match (<0.5)."""
    p = er.adjudicate_llm(ACME[0], ACME[3], SRCS)   # Acme Corp vs Acme Logistics
    print(f"\nadjudicate(Acme Corp, Acme Logistics) = {p:.3f}")
    assert p < 0.5


# ===========================================================================
# Tier 3 -- full pipeline with real embedder + real LLM (reproduces Example 5)
# ===========================================================================
@needs_llm
def test_real_cluster_entities_runs_and_merges_family():
    """End to end with real embeddings + a real gpt-5-nano adjudicator.

    Asserts the ROBUST guarantees:
      * the pipeline runs and assigns every mention an entity_id, and
      * the near-identical pair x1/x2 resolves into one entity.

    It does NOT assert that x4 ('Acme Logistics') lands in its own entity. With
    the prototype's connected-components clustering (guide 8.5), ONE stray
    positive edge collapses the whole block, and gpt-5-nano is an imperfect
    adjudicator -- so x4 separation depends on the model being consistent across
    every band pair, which isn't guaranteed. The separation GUARANTEE is tested
    at the adjudicator level (test_real_adjudicator_rejects_obvious_nonmatch).
    If you want clustering itself to split x4 reliably, upgrade cluster_entities
    to the confident-split refinement described in guide 8.5.
    """
    clusters = er.cluster_entities(ACME, er.Matcher(), SRCS,
                                   adjudicator=er.adjudicate_llm)
    groups: dict[str, set[str]] = {}
    for mid, eid in clusters.items():
        groups.setdefault(eid, set()).add(mid)
    print("\nclusters:", {eid: sorted(ids) for eid, ids in groups.items()})

    assert set(clusters) == {"x1", "x2", "x3", "x4"}      # everyone resolved
    assert clusters["x1"] == clusters["x2"]               # near-identical -> merged

    # diagnostic only (not an assertion): did the family resolve cleanly?
    if clusters["x1"] != clusters["x4"]:
        print("x4 correctly separated -> full Example 5 reproduced")
    else:
        print("NOTE: x4 over-merged into the Acme entity. This is the known "
              "connected-components over-merge (guide 8.5) under a noisy "
              "small-model adjudicator -- not a matcher bug; the adjudicator "
              "itself rejects x1-x4 (see the adjudicator test).")
