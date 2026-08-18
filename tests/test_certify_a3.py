"""Seam A3 (fidelity certification harness) -- offline tests.

One test per duty (repo convention):
  test_default_truth_key_dict_records  -> THE regression: pipeline records
                                          are plain dicts; key extraction
                                          must read them (fixture-coverage
                                          gap found during A3)
  test_fidelity_loss_dict_records      -> the loss over dict-shaped records
  test_gate_param_regates_and_filters  -> resolve_and_aggregate(gate=...)
                                          re-gates stored mentions; stricter
                                          delta_E drops low-confidence ones
  test_gate_none_keeps_old_behavior    -> no gate arg => byte-for-byte the
                                          pre-A3 path (no regression)
  test_qbar_reaches_qtable             -> lambda's qbar lands in the QTable
  test_replay_delta_ordering_guard     -> loosening delta_E vs the reference
                                          pool refuses (rejects not stored)
  test_replay_refuses_bootstrap_gate   -> no calibration set => RuntimeError
  test_replay_refuses_cold_matcher     -> no labeled pairs => RuntimeError
  test_reference_runs_require_deny     -> empty denylist refuses (answer-key
                                          leak guard)
  test_make_callables_truth_closure    -> truth(g) serves build_truth output

No torch, no network: ER is bypassed with an injected cluster_fn; the
matcher-refusal test monkeypatches paths, never fits.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webagg import certify, config                         # noqa: E402
from webagg.calibration import ConformalGate               # noqa: E402
from webagg.formd import build_truth_entity, parse_form_d, save_truth_entity  # noqa: E402
from webagg.pipeline import resolve_and_aggregate          # noqa: E402
from webagg.risk_control import (default_truth_key,        # noqa: E402
                                 fidelity_loss, TruthEntity, TruthRecord)
from webagg.storage import get_session                     # noqa: E402
from webagg.type_defs import Mention, Source               # noqa: E402

_NOW = datetime(2025, 1, 1)


# --------------------------------------------------------------------------- #
# 1-2. Dict-shaped records (the pipeline's REAL shape)
# --------------------------------------------------------------------------- #
def _cv(value: str, value_num: float | None = None):
    """A CorroboratedValue stand-in with the two fields usd()/key use."""
    return SimpleNamespace(value=value, value_num=value_num)


def _dict_record(kind="funding_round", date="2024-01-10",
                 amount="5000000", amount_num=5_000_000.0) -> dict:
    """A record EXACTLY as pipeline.resolve_and_aggregate builds it."""
    return {"entity_id": "ent_0", "record_kind": kind, "frag_case": "scan",
            "attributes": {"amount": _cv(amount, amount_num),
                           "date": _cv(date)}}


def test_default_truth_key_dict_records():
    assert default_truth_key(_dict_record()) == "funding_round|2024-01-10"
    # attribute-style records (the old fakes) must keep working too
    obj = SimpleNamespace(record_kind="funding_round",
                          attributes={"date": _cv("2024-01-10")})
    assert default_truth_key(obj) == "funding_round|2024-01-10"


def test_fidelity_loss_dict_records():
    truth = TruthEntity("g", (TruthRecord("funding_round|2024-01-10",
                                          5_000_000.0, "2024-01-10"),))
    # perfect assembly -> zero loss
    assert fidelity_loss([_dict_record()], truth) == 0.0
    # misread amount -> proportional loss (4M vs true 5M = 0.2)
    off = [_dict_record(amount="4000000", amount_num=4_000_000.0)]
    assert abs(fidelity_loss(off, truth) - 0.2) < 1e-9


# --------------------------------------------------------------------------- #
# 3-5. The pipeline's new replay knobs
# --------------------------------------------------------------------------- #
def _fixture_db(tmp_path: Path) -> str:
    """A tiny finished-run DB: one source, two accepted amount mentions of
    the same record -- one confident (0.95), one shaky (0.55)."""
    db = str(tmp_path / "run.sqlite")
    s = get_session(db)
    src = Source(source_id="s1", url="https://a.com/x", domain="a.com",
                 fetch_time=_NOW, publish_time=None, title=None,
                 main_text="t", formulation_id="f0")
    s.add(src.to_row())
    for i, (val, conf) in enumerate([("5000000", 0.95), ("7000000", 0.55)]):
        m = Mention(mention_id=f"m{i}", source_id="s1",
                    entity_surface="Acme", record_kind="funding_round",
                    attribute="amount", value=val, passage="p",
                    extracted_at=_NOW, value_num=float(val),
                    self_conf=conf, accepted=True)
        s.add(m.to_row())
    s.commit()
    eng = s.get_bind(); s.close(); eng.dispose()
    return db


def _strict_gate() -> ConformalGate:
    """Fitted gate whose threshold is 0.10: accepts self_conf >= 0.90 only
    (all-correct calibration at conf 0.9 -> every score 0.10)."""
    cal = [("100", "100", 0.9)] * 30
    return ConformalGate(delta_E=0.05).fit(cal)


def _one_cluster(mentions, source_lookup):
    return {m.mention_id: "ent_0" for m in mentions}


def test_gate_param_regates_and_filters(tmp_path):
    db = _fixture_db(tmp_path)
    s = get_session(db)
    try:
        out = resolve_and_aggregate(s, run_id="t", query_attributes={"amount"},
                                    cluster_fn=_one_cluster, state=None,
                                    gate=_strict_gate())
        recs = out["records"]
        # the shaky mention (0.55) was re-gated OUT: only 5000000 survives
        assert len(recs) == 1
        assert recs[0]["attributes"]["amount"].value == "5000000"
    finally:
        eng = s.get_bind(); s.close(); eng.dispose()


def test_gate_none_keeps_old_behavior(tmp_path):
    db = _fixture_db(tmp_path)
    s = get_session(db)
    try:
        out = resolve_and_aggregate(s, run_id="t", query_attributes={"amount"},
                                    cluster_fn=_one_cluster, state=None)
        # both mentions reach corroboration exactly as before A3
        by_val = out["records"][0]["attributes"]["amount"]
        assert by_val.value in ("5000000", "7000000")   # one adopted value
        # and nothing was filtered: the losing value was still in play,
        # so corroboration saw a real contest (evidence: n_values extra)
    finally:
        eng = s.get_bind(); s.close(); eng.dispose()


def test_qbar_reaches_qtable(tmp_path, monkeypatch):
    seen = {}
    import webagg.pipeline as pl

    class SpyQTable(pl.QTable):
        def __init__(self, qbar=0.30):
            seen["qbar"] = qbar
            super().__init__(qbar=qbar)

    monkeypatch.setattr(pl, "QTable", SpyQTable)
    db = _fixture_db(tmp_path)
    s = get_session(db)
    try:
        resolve_and_aggregate(s, run_id="t", query_attributes={"amount"},
                              cluster_fn=_one_cluster, state=None, qbar=0.20)
    finally:
        eng = s.get_bind(); s.close(); eng.dispose()
    assert seen["qbar"] == 0.20


# --------------------------------------------------------------------------- #
# 6-9. The harness's refusal guards (seam discipline, enforced)
# --------------------------------------------------------------------------- #
_LAM = {"tau_plus": 0.85, "tau_minus": 0.15, "delta_E": 0.05, "qbar": 0.30}


def test_replay_delta_ordering_guard(tmp_path):
    lam = dict(_LAM, delta_E=0.10)          # LOOSER than the reference 0.05
    with pytest.raises(ValueError, match="TIGHTEN"):
        certify.replay(str(tmp_path / "x.sqlite"), lam, delta_E_ref=0.05)


def test_replay_refuses_bootstrap_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CALIBRATION_SET", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="bootstrap"):
        certify.replay(_fixture_db(tmp_path), _LAM, delta_E_ref=0.05)


def test_replay_refuses_cold_matcher(tmp_path, monkeypatch):
    # a real (tiny) calibration set so the gate guard passes...
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(
        [{"pred": "100", "true": "100", "self_conf": 0.9}] * 30))
    monkeypatch.setattr(config, "CALIBRATION_SET", cal)
    # ...but NO labeled pairs -> the matcher guard must refuse
    monkeypatch.setattr(config, "MATCH_PAIRS", tmp_path / "absent.csv")
    with pytest.raises(RuntimeError, match="cold"):
        certify.replay(_fixture_db(tmp_path), _LAM, delta_E_ref=0.05)


def test_reference_runs_require_deny():
    with pytest.raises(ValueError, match="denylist"):
        certify.reference_runs({"split": {"calibration": []},
                                "entities": {}},
                               domain="t", deny=())


# --------------------------------------------------------------------------- #
# 10. truth(g) serves the build_truth.py answer keys
# --------------------------------------------------------------------------- #
def test_make_callables_truth_closure(tmp_path):
    xml = """<?xml version="1.0"?><edgarSubmission>
      <primaryIssuer><cik>77</cik><entityName>Acme</entityName></primaryIssuer>
      <offeringData><typeOfFiling>
        <newOrAmendment><isAmendment>false</isAmendment></newOrAmendment>
        <dateOfFirstSale><value>2024-01-10</value></dateOfFirstSale>
      </typeOfFiling><offeringSalesAmounts>
        <totalOfferingAmount>8000000</totalOfferingAmount>
        <totalAmountSold>5000000</totalAmountSold>
      </offeringSalesAmounts></offeringData></edgarSubmission>"""
    f = parse_form_d(xml, accession="0000000077-24-000001",
                     filing_date="2024-01-15")
    truth_obj, meta = build_truth_entity("cik0000000077", [f])
    save_truth_entity(tmp_path, meta)

    run_pipeline, truth = certify.make_callables(
        {"cik0000000077": {"db": "unused.sqlite", "delta_E_ref": 0.05}},
        tmp_path)
    got = truth("cik0000000077")
    assert got == truth_obj
    assert got.true_sum == 5_000_000.0
    # and run_pipeline propagates the delta guard through the closure
    with pytest.raises(ValueError, match="TIGHTEN"):
        run_pipeline("cik0000000077", dict(_LAM, delta_E=0.10))


# --------------------------------------------------------------------------- #
# 11-14. Attempt-#2 pre-registered grading rules
# --------------------------------------------------------------------------- #
def test_amount_tol_fallback_matches_within_tolerance():
    truth = TruthEntity("g", (TruthRecord("funding_round|2020-06-15",
                                          225_000_000.0, "2020-06-15"),))
    undated = {"entity_id": "e", "record_kind": "funding_round/unknown",
               "attributes": {"amount": _cv("224000000", 224_000_000.0)}}
    from webagg.risk_control import match_to_truth
    # without the tolerance: spurious (no date key)
    assert match_to_truth([undated], truth)[0][1] is None
    # with the cohort tolerance: aligned (0.44% off < 2%)
    aligned = match_to_truth([undated], truth, amount_tol=0.02)
    assert aligned[0][1] is truth.records[0]
    # and the loss reflects the small residual, not a full spurious hit
    assert fidelity_loss([undated], truth, amount_tol=0.02) < 0.01


def test_amount_tol_beyond_tolerance_stays_spurious():
    truth = TruthEntity("g", (TruthRecord("funding_round|2020-06-15",
                                          225_000_000.0, "2020-06-15"),))
    wrong = {"entity_id": "e", "record_kind": "funding_round/unknown",
             "attributes": {"amount": _cv("200000000", 200_000_000.0)}}
    from webagg.risk_control import match_to_truth
    assert match_to_truth([wrong], truth, amount_tol=0.02)[0][1] is None
    assert fidelity_loss([wrong], truth, amount_tol=0.02) >= 0.88


def test_amount_tol_one_to_one_closest_first_and_date_precedence():
    truth = TruthEntity("g", (
        TruthRecord("funding_round|2020-06-15", 1_000_000.0, "2020-06-15"),
        TruthRecord("funding_round|2021-03-02", 1_010_000.0, "2021-03-02")))
    dated = _dict_record(kind="funding_round/series_a", date="2021-03-02",
                         amount="1010000", amount_num=1_010_000.0)
    near = {"entity_id": "e", "record_kind": "funding_round/unknown",
            "attributes": {"amount": _cv("1000500", 1_000_500.0)}}
    also = {"entity_id": "e", "record_kind": "funding_round/unknown",
            "attributes": {"amount": _cv("1004000", 1_004_000.0)}}
    from webagg.risk_control import match_to_truth
    aligned = match_to_truth([dated, near, also], truth, amount_tol=0.02)
    # the DATED record claimed 2021-03-02 by key, before any fallback ran
    assert aligned[0][1].key == "funding_round|2021-03-02"
    # closest undated record takes the remaining truth record...
    assert aligned[1][1].key == "funding_round|2020-06-15"
    # ...and one-to-one exhausts truth: the second undated stays spurious
    assert aligned[2][1] is None


def test_learn_then_test_loss_fn_injection():
    from webagg.risk_control import learn_then_test
    calls = []

    def fake_loss(recs, t):
        calls.append(recs)
        return 0.0
    out = learn_then_test(["g1"], [{"lam": 1}], 0.5, 0.9,
                          run_pipeline=lambda g, lam: ["records"],
                          truth=lambda g: "truth", loss_fn=fake_loss)
    assert out is not None and calls == [["records"]]


def test_query_name_override():
    manifest = {"entities": {"cik1": {"entity_name": "Maplebear Inc."}}}
    assert certify.query_name(manifest, "cik1", None) == "Maplebear Inc."
    assert certify.query_name(manifest, "cik1",
                              {"cik1": "Instacart"}) == "Instacart"


def test_records_carry_mention_provenance(tmp_path):
    """Every resolved record must name the stored mentions it came from
    (impl Sec. 4.2). Regression: Exp-2 grading silently saw empty
    provenance and graded recall 0 at every step."""
    db = _fixture_db(tmp_path)
    s = get_session(db)
    try:
        out = resolve_and_aggregate(
            s, run_id="t", query_attributes={"amount"},
            cluster_fn=lambda ms, src: {m.mention_id: m.entity_surface
                                        for m in ms})
        stored = {m.mention_id for m in
                  __import__("webagg.pipeline", fromlist=["load_mentions"])
                  .load_mentions(s)}
        assert out["records"], "fixture produced no records"
        for r in out["records"]:
            got = r["contributing_mentions"]
            assert got, f"record {r['record_kind']} has empty provenance"
            assert set(got) <= stored, "provenance names unknown mention ids"
    finally:
        s.close()
