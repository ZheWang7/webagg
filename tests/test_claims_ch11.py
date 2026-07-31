"""§11 smoke tests -- the Claims Engine: checksums and gap-directed search.

Guide's test spec, one test per bullet:
  * SUM certifies within tolerance (12 + 40 + 11 = 63, small Delta+);
  * COUNT deficit (claim 3, found 2) blocks stop;
  * a gap formulation for value_gap = 11e6, eras = [2021, 2025] mentions
    "$11" and a 2019-ish year;
  * scope "including debt" never certifies;
  * found 4 vs. claim 3 raises a conflict.
Plus the three discipline rules made testable: the certifying belief is
STORED (R1), demotions are LOGGED (R2), and a Chao-vs-checksum disagreement
withdraws the cert and pushes a verification item (R3). And the §11-specific
corroboration behaviors: as_of supersession along an annual-report chain,
and the LLM-free fallback formulation.

All offline: the one LLM seam (gap_formulations) is monkeypatched, which
tests the PLUMBING (gap payload in, Formulation fields out, insert-once
dedup) -- the live prompt's phrasing quality is a sanity-run concern.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg.claims import ClaimsEngine, CoverageView, implied_value_tolerance  # noqa: E402
from webagg.frontier import FrontierState, StratumState                        # noqa: E402
from webagg.type_defs import Claim, Source                                     # noqa: E402
import webagg.claims as claims_mod                                             # noqa: E402


# ---------------------------------------------------------------------------
# fixtures: a session stub, sources, claims
# ---------------------------------------------------------------------------

class _StubSession:
    def merge(self, row):          # ingest() persists; these tests don't
        pass


def _mk_source(sid, domain, text, *, anchored=False, source_class="news",
               chain=None, doc_type=None, published=None) -> Source:
    """Minimal Source. Distinct main_texts (no shared shingles, no domain
    cross-references) keep the derivation graph edge-free, so each source
    is its own independent component."""
    return Source(
        source_id=sid, url=f"https://{domain}/x", domain=domain,
        fetch_time=datetime(2026, 1, 1),                # UTC-naive, project-wide
        publish_time=published or datetime(2025, 6, 1),
        title=None, main_text=text, formulation_id="f0",
        source_class=source_class, identity_anchored=anchored,
        authority_chain_id=chain, doc_type=doc_type)


def _mk_claim(sid, functional, value, *, g="acme", scope="", tol=0.0,
              t_asof=None, passage="") -> Claim:
    return Claim(
        claim_id=Claim.make_id(sid, functional, g), source_id=sid,
        stratum_surface=g, functional=functional, attribute="amount",
        value_num=float(value), currency="USD", t_asof=t_asof,
        scope=scope, tolerance=tol, passage=passage)


def _engine_with_sources(*sources) -> ClaimsEngine:
    ce = ClaimsEngine(_StubSession())
    for s in sources:
        ce.register_source(s)
    return ce


TEXT_A = "Acme closed its Series B, a sixty three million dollar journey so far."
TEXT_B = "Robotics startup Acme now counts three completed equity financings."


# ---------------------------------------------------------------------------
# Theorem 4(a): SUM certifies within tolerance (12 + 40 + 11 = 63)
# ---------------------------------------------------------------------------

def test_sum_certifies_within_tolerance_and_stores_belief():
    ce = _engine_with_sources(_mk_source("s1", "techdaily.example", TEXT_A),
                              _mk_source("s2", "vcwire.example", TEXT_B))
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "SUM", 63e6,
                            scope="three equity rounds", tol=0.5e6))
    # assembled table: 12M + 40M + 11M = 63M; small E_g from stated precision
    st = ce.checksum("acme", CoverageView(n_records=3, sum=63e6, E_g=0.3e6))
    assert st.certified and st.kind == "SUM"
    # Delta+ = max(0, 63e6 - 63e6) + t_V + E_g = 0.8e6 <= 0.02 * 63e6
    assert st.delta_plus == 0.5e6 + 0.3e6
    # rule R1: certification is CONDITIONAL and says so -- two independent
    # unanchored origins, each capped at qbar=0.3: b = 1 - 0.7^2 = 0.51
    assert abs(st.belief - 0.51) < 1e-9


# ---------------------------------------------------------------------------
# Theorem 4(b): COUNT deficit (claim 3, found 2) blocks stop
# ---------------------------------------------------------------------------

def test_count_deficit_blocks_stop():
    ce = ClaimsEngine(_StubSession())          # no sources: degenerate path
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "COUNT", 3))
    st = ce.checksum("acme", CoverageView(n_records=2))
    assert not st.certified
    assert st.gap == {"count_gap": 1}          # the residual, as a steer
    # ...and the App. E hard brake arms via the CORROBORATED count (belief
    # 0.51 >= CLAIM_BRAKE_MIN_BELIEF); all_strata_pass (ch. 7) refuses to
    # stop while N_g < claimed_count.
    state = FrontierState()
    state.strata["acme"] = StratumState(name="acme")
    ce.update_cardinality_brakes(state)
    assert state.strata["acme"].claimed_count == 3
    # a single witness does NOT arm it (belief 0.3 < 0.5)
    ce2 = ClaimsEngine(_StubSession())
    ce2.ingest(_mk_claim("s1", "COUNT", 3))
    state2 = FrontierState()
    state2.strata["acme"] = StratumState(name="acme")
    ce2.update_cardinality_brakes(state2)
    assert state2.strata["acme"].claimed_count is None


# ---------------------------------------------------------------------------
# Definition 12: the residual is a query (LLM seam stubbed -> plumbing)
# ---------------------------------------------------------------------------

def test_gap_formulation_uses_constraints_and_dedups(monkeypatch):
    seen = {}

    def fake_llm(*, system, user, **kw):
        import json
        p = json.loads(user)
        seen.update(p)
        # a well-behaved LLM uses every constraint: dollar magnitude from
        # value_gap, a year BELOW the covered eras, the entity name
        q = f'{p["entity"]} seed 2019 "${p["value_gap"] / 1e6:.0f} million"'
        return {"payload": {"formulations": [
            {"query": q, "p_success": 0.7, "yield_if_success": 1}]},
            "input_tokens": 10, "output_tokens": 10, "model": "fake"}

    monkeypatch.setattr(claims_mod, "call_llm", fake_llm)
    ce = ClaimsEngine(_StubSession())
    gap = {"value_gap": 11e6, "eras_covered": [2021, 2025], "stages": ["series b"]}
    fs = ce.gap_formulations("acme", gap)
    assert seen["value_gap"] == 11e6 and seen["eras_covered"] == [2021, 2025]
    assert len(fs) == 1 and fs[0].gap_directed and fs[0].stratum == "acme"
    assert "$11" in fs[0].query and "2019" in fs[0].query
    # unusually high yield estimate: it searches for something KNOWN to exist
    assert fs[0].p_success == 0.7
    # insert-once per (stratum, gap signature): the engine re-detects the
    # same gap every step it persists -- must not re-emit (guide §7.3)
    assert ce.gap_formulations("acme", gap) == []


def test_gap_formulation_falls_back_without_llm(monkeypatch):
    def broken_llm(**kw):
        raise RuntimeError("offline")
    monkeypatch.setattr(claims_mod, "call_llm", broken_llm)
    ce = ClaimsEngine(_StubSession())
    fs = ce.gap_formulations("acme", {"count_gap": 2})
    assert len(fs) == 1 and fs[0].gap_directed     # the ch. 7 template
    assert fs[0].yield_if_success == 2.0


# ---------------------------------------------------------------------------
# Rule R2: scope mismatches demote, never certify -- and are LOGGED
# ---------------------------------------------------------------------------

def test_scope_including_debt_never_certifies():
    ce = _engine_with_sources(_mk_source("s1", "techdaily.example", TEXT_A),
                              _mk_source("s2", "vcwire.example", TEXT_B))
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "SUM", 80e6, scope="including debt",
                            tol=0.5e6))
    # the table even MATCHES the demoted total -- still no cert: "raised
    # $80M including debt" must not certify an equity-rounds stratum
    st = ce.checksum("acme", CoverageView(n_records=3, sum=80e6, E_g=0.1e6))
    assert not st.certified
    assert ce.demotion_rate == 1.0 and len(ce.demotions) == 2   # logged


# ---------------------------------------------------------------------------
# Rule R3: conflicts outrank both sides
# ---------------------------------------------------------------------------

def test_found_more_than_claimed_raises_conflict():
    ce = ClaimsEngine(_StubSession())
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "COUNT", 3))
        ce.ingest(_mk_claim(sid, "SUM", 63e6, tol=0.5e6))
    st = ce.checksum("acme", CoverageView(n_records=4, sum=63e6, E_g=0.1e6))
    assert st.conflict and not st.certified
    # ...and the matching SUM must NOT paper it over (conflicts outrank)
    assert any(c["kind"] == "found_more_than_claimed" for c in ce.conflicts)


def test_chao_vs_checksum_withdraws_certification():
    ce = ClaimsEngine(_StubSession())
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "COUNT", 3))
    # claim says done (3 == 3) BUT Chao screams remainder -> flag, don't trust
    st = ce.checksum("acme", CoverageView(n_records=3, chao_m0=3.0))
    assert not st.certified and st.conflict
    assert any(c["kind"] == "chao_vs_checksum" for c in ce.conflicts)
    # with a quiet Chao the same view certifies cleanly (Thm 4b, recall 1)
    st2 = ClaimsEngine(_StubSession())
    for sid in ("s1", "s2"):
        st2.ingest(_mk_claim(sid, "COUNT", 3))
    assert st2.checksum("acme", CoverageView(n_records=3)).certified


# ---------------------------------------------------------------------------
# Claim supersession: a newer annual report supersedes last year's total
# ---------------------------------------------------------------------------

def test_asof_supersession_along_annual_report_chain():
    chain = "acme:annual"
    ce = _engine_with_sources(
        _mk_source("fy24", "acme.example", TEXT_A, chain=chain,
                   doc_type="annual_report", published=datetime(2025, 2, 1)),
        _mk_source("fy25", "vcwire.example", TEXT_B, chain=chain,
                   doc_type="annual_report", published=datetime(2026, 2, 1)))
    ce.ingest(_mk_claim("fy24", "SUM", 63e6, tol=0.5e6,
                        t_asof=datetime(2024, 12, 31)))
    ce.ingest(_mk_claim("fy25", "SUM", 75e6, tol=0.5e6,
                        t_asof=datetime(2025, 12, 31)))
    v, b, tol = ce.corroborated("acme", "SUM")
    assert v == 75e6            # the FY2025 total superseded FY2024's,
    assert b > 0                # not outvoted -- disqualified by as_of


# ---------------------------------------------------------------------------
# E_g helper: stated precision, conservatively capped
# ---------------------------------------------------------------------------

def test_implied_value_tolerance():
    assert implied_value_tolerance(0) == 0.0
    assert implied_value_tolerance(12_000_000) == 0.12e6     # 1% cap binds
    assert implied_value_tolerance(1_500) == 15.0            # 1% cap binds
    assert implied_value_tolerance(101) == 0.5               # half last digit
