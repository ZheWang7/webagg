"""§14 smoke tests -- End-to-End Pipeline: the two-term honest interval,
the second (post-ER) checksum pass with REVOCATION, the verification
allocator, the optional certified-refine, and the CLI table.

One test per edit, per repo discipline:
  * report.py     -- every eps_C regime (registry / COUNT / SUM /
                     statistical / ABANDONED), the value-weighted global
                     combine, and the eps_F provenance chain
                     (given > ltt certificate > labelled fallback);
  * pipeline.py   -- revalidate_certificates(): REVOKE on a post-ER count
                     gap, REVOKE + veto on a fragile-pair conflict,
                     CERTIFY post-ER when claims meet only at the resolved
                     key (measurement names preserved for the ch.-11
                     online sanity);
  * verify.py     -- conflicts first at infinite drop, greedy ordering by
                     width removed, the budget cap, weak_entity_link and
                     supersession cells picked up;
  * corroboration -- refine_from_certified() moves an agreeing source up
                     and a disagreeing one down (Beta posterior), while
                     the qbar cap still binds unanchored sources; the
                     default path (flag off) is untouched;
  * report.py     -- format_report() prints a PER-GROUP table (the guide's
                     acceptance test: never one number with one interval).

All offline: states, engines and records are hand-built; the ClaimsEngine
runs on its degenerate stub-session path (two unanchored witnesses ->
belief 0.51, same convention as the ch. 11 tests).
"""
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg import config                                           # noqa: E402
from webagg.claims import ClaimsEngine                              # noqa: E402
from webagg.corroboration import QTable, refine_from_certified      # noqa: E402
from webagg.frontier import FrontierState, StratumState             # noqa: E402
from webagg.pipeline import revalidate_certificates                 # noqa: E402
from webagg.report import (aggregate_two_term, format_report,       # noqa: E402
                           resolve_eps_F)
from webagg.type_defs import Claim, CorroboratedValue, Mention, Source  # noqa: E402
from webagg.verify import verification_menu                         # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class _StubSession:
    def merge(self, row):          # ingest() persists; these tests don't
        pass


def _cv(value_num, belief, *, kappa=None, flags=(), n_dead=0) -> CorroboratedValue:
    """Minimal adopted value for report/verify tests."""
    return CorroboratedValue(
        value=str(value_num), value_num=float(value_num), belief=belief,
        nu=1, component_sizes=[1], kappa=kappa,
        validator_flags=list(flags), n_dead_excluded=n_dead)


def _rec(eid, amount_cv, kind="funding_round") -> dict:
    return {"entity_id": eid, "record_kind": kind, "frag_case": "scan",
            "attributes": {"amount": amount_cv}}


def _mk_claim(sid, functional, value, *, g="acme", scope="two equity rounds",
              tol=0.0) -> Claim:
    return Claim(
        claim_id=Claim.make_id(sid, functional, g), source_id=sid,
        stratum_surface=g, functional=functional, attribute="amount",
        value_num=float(value), currency="USD", scope=scope, tolerance=tol)


def _mention(mid, sid, value_num, *, surface="Acme") -> Mention:
    return Mention(
        mention_id=mid, source_id=sid, entity_surface=surface,
        record_kind="funding_round", attribute="amount",
        value=f"${value_num:,.0f}", value_num=float(value_num),
        passage="p", extracted_at=datetime(2026, 1, 1))


def _state_with(*strata: StratumState) -> FrontierState:
    st = FrontierState()
    for s in strata:
        st.strata[s.name] = s
    return st


def _add_records(state, g, n, occasions_each=2):
    """Register n records in stratum g, each seen `occasions_each` times.
    occasions_each >= 2 -> no singletons -> U_hat = 0 (no formulations)."""
    for i in range(n):
        rk = f"{g}|rec{i}"
        state.record_stratum[rk] = g
        state.covered[rk] = {f"f{i}_{j}" for j in range(occasions_each)}


# ---------------------------------------------------------------------------
# §14.2 -- the two-term interval, every regime
# ---------------------------------------------------------------------------

def test_two_term_interval_all_regimes():
    state = _state_with(
        StratumState(name="count_co", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.9),
        StratumState(name="sum_co", certified="checksum", cert_kind="SUM",
                     cert_belief=0.8, cert_delta_plus=1e6),
        StratumState(name="stat_co", V=1e-8),      # tiny realized V -> tiny psi
        StratumState(name="aband_co"),
    )
    _add_records(state, "count_co", 2)
    _add_records(state, "sum_co", 2)
    _add_records(state, "stat_co", 3)                        # doubletons: U_hat=0
    _add_records(state, "aband_co", 1, occasions_each=1)     # a singleton: U_hat=1

    resolved = [
        _rec("count_co", _cv(10e6, 0.9, kappa=3)),
        _rec("count_co", _cv(5e6, 0.8, kappa=1)),
        _rec("sum_co", _cv(40e6, 0.9, kappa=2)),
        _rec("sum_co", _cv(10e6, 0.9, kappa=2)),
        _rec("stat_co", _cv(20e6, 0.9, kappa=4)),
        _rec("aband_co", _cv(7e6, 0.6)),
    ]
    rep = aggregate_two_term(resolved, state, eps_g=0.10, eps_F=0.05,
                             delta_M=0.10, max_steps=200)

    # COUNT: eps_C = 1 - belief; kappa = the group's WEAKEST cell
    row = rep["count_co"]
    assert math.isclose(row["eps_C"], 0.1) and row["min_kappa"] == 1
    assert math.isclose(row["halfwidth"], (0.1 + 0.05) * 15e6)

    # SUM: eps_C = Delta+/total + (1 - belief)
    row = rep["sum_co"]
    assert math.isclose(row["eps_C"], 1e6 / 50e6 + 0.2)

    # statistical: U_hat + psi < eps_g at report time -> eps_C = eps_g
    row = rep["stat_co"]
    assert row["regime"] == "STATISTICAL" and math.isclose(row["eps_C"], 0.10)

    # ABANDONED: no interval, the achieved U_hat + psi instead (>= 1 here:
    # the lone singleton alone gives U_hat = 1)
    assert "aband_co" not in rep and "aband_co" in rep["__abandoned__"]
    ab = rep["__abandoned__"]["aband_co"]
    assert ab["U_plus_psi"] >= 1.0 and ab["total"] == 7e6

    # global: value-weighted eps_C over CERTIFIED groups only; halfwidths add
    gl = rep["__global__"]
    assert gl["total"] == 85e6                       # abandoned 7e6 excluded
    want = (0.1 * 15e6 + (1e6 / 50e6 + 0.2) * 50e6 + 0.1 * 20e6) / 85e6
    assert math.isclose(gl["eps_C"], want)
    assert math.isclose(gl["halfwidth"],
                        sum(rep[g]["halfwidth"]
                            for g in ("count_co", "sum_co", "stat_co")))


def test_cardinality_brake_forbids_statistical_pass():
    # U_hat + psi would pass, but a corroborated COUNT claim says a record
    # is still missing -> App. E brake -> ABANDONED, not statistical.
    S = StratumState(name="acme", V=1e-8, claimed_count=3)
    state = _state_with(S)
    _add_records(state, "acme", 2)
    rep = aggregate_two_term([_rec("acme", _cv(1e6, 0.9))], state,
                             eps_g=0.10, eps_F=0.05)
    assert "acme" in rep["__abandoned__"]


def test_registry_regime_schema_mode():
    # Theorem 3: schema sweep -> zero completeness slack, no state needed.
    rep = aggregate_two_term([_rec("acme", _cv(10e6, 0.95, kappa=5))], None,
                             mode="schema", eps_F=0.05)
    row = rep["acme"]
    assert row["eps_C"] == 0.0 and "registry" in row["certificate"]
    assert math.isclose(row["halfwidth"], 0.05 * 10e6)


def test_eps_F_provenance_chain(monkeypatch, tmp_path):
    # explicit override wins outright
    assert resolve_eps_F(eps_F=0.07) == (0.07, "given")
    # a stored §13 certificate for the domain -> "ltt"
    import webagg.risk_control as rc
    monkeypatch.setattr(rc, "load_fidelity_cert", lambda d: 0.08)
    assert resolve_eps_F(domain="startup_funding") == (0.08, "ltt")
    # no certificate, no override -> the LABELLED fallback constant (§14's
    # decision: print an interval, but never impersonate a calibration)
    monkeypatch.setattr(rc, "load_fidelity_cert", lambda d: None)
    val, src = resolve_eps_F(domain="never_calibrated")
    assert val == config.EPS_F_FALLBACK and src == "fallback"


# ---------------------------------------------------------------------------
# §14.1 -- the second checksum pass: REVOKE / veto / certify post-ER
# ---------------------------------------------------------------------------

def _engine_with_count_claim(n=3, g="acme") -> ClaimsEngine:
    """Degenerate-path engine (stub session, no Sources): two unanchored
    witnesses of 'COUNT = n' -> belief 0.51, over the brake threshold."""
    ce = ClaimsEngine(_StubSession())
    for sid in ("s1", "s2"):
        ce.ingest(_mk_claim(sid, "COUNT", n, g=g))
    return ce


def test_revalidation_revokes_on_post_er_count_gap():
    # Discovery certified COUNT=3 over pre-ER surface records; ER then
    # merged two of them -> only 2 resolved records remain. The §14.1
    # second pass must catch it: the certificate was wrong -> REVOKED.
    S = StratumState(name="acme", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.51)
    state = _state_with(S)
    _add_records(state, "acme", 2)                  # post-rekey: N = 2 < 3
    by_record = {("acme", "funding_round"):
                 [_mention("m1", "s1", 12e6), _mention("m2", "s2", 40e6)]}
    revalidate_certificates(state, _engine_with_count_claim(3), {}, by_record)
    assert S.certified is None and S.cert_kind is None and S.cert_belief is None


def test_revalidation_revokes_and_vetoes_on_fragile_conflict():
    # Post-ER the count MATCHES (N = 3 = claim), but one in-stratum ER
    # decision sat in the matcher band: the match is one coin toss deep.
    # Rule R3: verify, never certify -- and a pre-ER certificate is revoked.
    S = StratumState(name="acme", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.51)
    state = _state_with(S)
    _add_records(state, "acme", 3)
    ce = _engine_with_count_claim(3)
    frag_by = {"acme": [("mA", "mB", 0.52)]}
    revalidate_certificates(state, ce, frag_by, {})
    assert S.certified is None                       # revoked
    assert ("acme", "mA", "mB", 0.52) in ce.verification_queue   # queued (R3)


def test_revalidation_certifies_post_er():
    # Claims filed under two surfaces only meet after re-keying: discovery
    # never saw them together, so the stratum EARNS its certificate here.
    S = StratumState(name="acme")                    # uncertified
    state = _state_with(S)
    _add_records(state, "acme", 3)                   # N = 3 = claim, no fragiles
    revalidate_certificates(state, _engine_with_count_claim(3), {}, {})
    assert S.certified == "checksum" and S.cert_kind == "COUNT"
    assert abs(S.cert_belief - 0.51) < 1e-9          # rule R1: belief stored


def test_revalidation_keeps_a_clean_certificate():
    # Nothing moved post-ER -> the certificate stands untouched.
    S = StratumState(name="acme", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.51)
    state = _state_with(S)
    _add_records(state, "acme", 3)
    revalidate_certificates(state, _engine_with_count_claim(3), {}, {})
    assert S.certified == "checksum" and S.cert_belief == 0.51


# ---------------------------------------------------------------------------
# §14.3 -- the verification allocator
# ---------------------------------------------------------------------------

def test_verification_menu_order_budget_and_kinds():
    state = _state_with(
        StratumState(name="acme", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.6))
    resolved = [
        _rec("acme", _cv(50e6, 0.5)),                          # low belief
        _rec("acme", _cv(9e6, 0.95, flags=["weak_entity_link"])),
        _rec("beta", _cv(30e6, 0.9, n_dead=2)),                # supersession
    ]
    rep = aggregate_two_term(resolved, state, eps_g=0.1, eps_F=0.05)
    ce = SimpleNamespace(conflicts=[{"why": "found>claimed"}],
                         verification_queue=[("acme", "mA", "mB", 0.5)])
    menu = verification_menu(resolved, ce, state, report=rep,
                             fragile_pairs=[("mA", "mB", 0.5),
                                            ("mC", "mD", 0.48)],
                             alpha=0.05, budget=5)
    assert len(menu) == 5                                       # budget cap
    # conflicts FIRST at infinite drop (rule R3 outranks everything) --
    # including the count-sensitivity pair escalated from the queue
    assert [c["kind"] for c in menu[:2]] == ["conflict", "conflict"]
    assert all(math.isinf(c["drop"]) for c in menu[:2])
    # the queued pair must not ALSO appear as a finite 'merge' duplicate
    assert not any(c["kind"] == "merge" and "mA" in c["what"] for c in menu)
    # the rest is greedy by width removed
    finite = [c["drop"] for c in menu[2:]]
    assert finite == sorted(finite, reverse=True)
    # the biggest finite item is the low-belief $50M cell: (1-0.5) * 50e6
    assert menu[2]["kind"] == "value" and math.isclose(menu[2]["drop"], 25e6)


def test_verification_menu_survives_fixture_path():
    # ce=None, state=None (legacy fixture path): no conflict/claim items,
    # cell-level checks still fire, nothing crashes.
    resolved = [_rec("acme", _cv(10e6, 0.4))]
    menu = verification_menu(resolved, None, None, report={}, budget=3)
    assert menu and menu[0]["kind"] == "value"


def test_verification_menu_survives_cold_start_alpha():
    # REGRESSION (found by the first live CLI run): on matcher cold start
    # ERResult.alpha is None -- the tracked uncertified-alpha seam -- and
    # the real path passes it straight through. The offline test above had
    # pinned alpha=0.05, hiding the None contract (the gap_formulations
    # lesson again: fakes must pin the REAL contract). Menu scoring must
    # fall back to its ranking default, never crash.
    resolved = [_rec("acme", _cv(10e6, 0.9))]
    rep = aggregate_two_term(resolved, None, eps_F=0.05)
    menu = verification_menu(resolved, None, None, report=rep,
                             fragile_pairs=[("mA", "mB", 0.5)],
                             alpha=None, budget=3)
    merges = [c for c in menu if c["kind"] == "merge"]
    assert merges and math.isclose(merges[0]["drop"], 0.05 * 10e6)


# ---------------------------------------------------------------------------
# §14.1 optional -- certified-refine (paper §4.4 / App. F)
# ---------------------------------------------------------------------------

def _src(sid, *, anchored, source_class="news") -> Source:
    return Source(source_id=sid, url=f"https://{sid}.example/x",
                  domain=f"{sid}.example", fetch_time=datetime(2026, 1, 1),
                  publish_time=datetime(2025, 6, 1),       # UTC-naive, project-wide
                  title=None, main_text="t", formulation_id="f0",
                  source_class=source_class, identity_anchored=anchored)


def test_refine_from_certified_moves_q_and_respects_cap():
    good = _src("good", anchored=True)     # agrees with certified values
    bad = _src("bad", anchored=True)       # disagrees
    capped = _src("capped", anchored=False)  # agrees, but no identity anchor
    lookup = {s.source_id: s for s in (good, bad, capped)}
    labeled = [("40000000",
                {"40000000": [_mention("a1", "good", 40e6),
                              _mention("a2", "capped", 40e6)],
                 "35000000": [_mention("d1", "bad", 35e6)]})] * 4  # 4 cells

    qt = QTable()
    q_prior = qt.q(good)                   # news prior 0.60
    n = refine_from_certified(qt, labeled, lookup, prior_strength=4.0)
    assert n == 3
    # agree-only source moves UP from its prior, disagree-only moves DOWN
    assert qt.q(good) > q_prior            # (4 + 4*0.6) / (4 + 4) = 0.8
    assert math.isclose(qt.q(good), 0.8)
    assert qt.q(bad) < q_prior             # (0 + 2.4) / 8 = 0.3
    # the qbar SECURITY cap still binds an unanchored source, however
    # much agreement it racked up -- agreement can be manufactured
    assert qt.q(capped) == qt.qbar


def test_refine_flag_off_default_path_untouched():
    # With no refinement recorded, q() is exactly the ch.-8 behavior.
    qt = QTable()
    assert qt.q(_src("s", anchored=True)) == 0.60          # news prior
    assert qt.q(_src("s", anchored=False)) == qt.qbar      # capped
    assert config.USE_CERTIFIED_REFINE is False            # default OFF


# ---------------------------------------------------------------------------
# §14.4 -- the CLI table: per-group rows, never one number
# ---------------------------------------------------------------------------

def test_format_report_prints_the_per_group_table():
    state = _state_with(
        StratumState(name="acme", certified="checksum",
                     cert_kind="COUNT", cert_belief=0.9),
        StratumState(name="zeta"),
    )
    _add_records(state, "acme", 2)
    _add_records(state, "zeta", 1, occasions_each=1)       # will abandon
    resolved = [_rec("acme", _cv(15e6, 0.9, kappa=2)),
                _rec("zeta", _cv(3e6, 0.7))]
    rep = aggregate_two_term(resolved, state, eps_g=0.1, eps_F=0.05)
    menu = verification_menu(resolved, None, state, report=rep, budget=2)
    txt = format_report(rep, menu, stop_reason="budget")

    assert "checksum COUNT (b=0.90)" in txt and "min kappa=2" in txt
    assert "ABANDONED (budget): achieved U+psi=" in txt    # honesty line
    assert "GLOBAL 15,000,000" in txt                      # certified only
    assert "fidelity eps_F=0.050 [given]" in txt           # provenance tag
    assert "Top human checks:" in txt
    # the guide's acceptance test: more than one interval-bearing row/line
    assert txt.count("+/-") >= 1 and "acme" in txt and "zeta" in txt


# ---------------------------------------------------------------------------
# thin-run regressions (shape of the FIRST successful live CLI run)
# ---------------------------------------------------------------------------

def test_all_abandoned_run_prices_merges_and_labels_global():
    """REGRESSION (first live run): one singleton stratum, nothing
    certified. Two things must hold: (1) fragile-pair merge checks are
    priced from the RESOLVED records' mean value, not the empty certified
    global (the live menu showed '-0' on every pair); (2) the global line
    says 'no certified strata', never a misleading 'GLOBAL 0 +/- 0'."""
    S = StratumState(name="ent_00000")
    state = _state_with(S)
    _add_records(state, "ent_00000", 1, occasions_each=1)   # lone singleton
    resolved = [_rec("ent_00000", _cv(30e9, 0.30))]         # one qbar witness
    rep = aggregate_two_term(resolved, state, eps_g=0.10, eps_F=0.15)
    assert rep["__global__"]["n"] == 0 and "ent_00000" in rep["__abandoned__"]

    menu = verification_menu(resolved, None, state, report=rep,
                             fragile_pairs=[("mA", "mB", 0.5)],
                             alpha=None, budget=5)
    merges = [c for c in menu if c["kind"] == "merge"]
    assert merges and math.isclose(merges[0]["drop"], 0.05 * 30e9)  # not -0
    # the top check is still the lone low-belief cell: (1-0.3)*30e9
    assert menu[0]["kind"] == "value" and math.isclose(menu[0]["drop"], 21e9)

    txt = format_report(rep, menu, stop_reason="max_steps")
    assert "GLOBAL: no certified strata (all ABANDONED)" in txt
    assert "GLOBAL 0" not in txt
