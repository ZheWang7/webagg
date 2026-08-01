"""Ch. 13 offline tests: the fidelity certificate (guide §13, paper §7 + App. H).

One test per duty, so a failure points at the responsible piece:
  * usd(): value_num preferred, canonicalizer fallback, unreadable -> 0
  * match_to_truth(): one-to-one alignment; over-splits punished; missing
    truth records excluded (they are completeness, not fidelity)
  * fidelity_loss(): the guide's exact L_g arithmetic -- misread amounts,
    spurious records at full value, the [0,1] cap, missing-record exclusion
  * hoeffding_p(): exact formula, the mean >= eps_F short-circuit, and
    monotonicity in n (more entities => stronger evidence)
  * learn_then_test(): the fixed-sequence discipline -- certify the passing
    prefix, STOP at the first failure, never evaluate configs beyond it
    (practical rule 1 / pitfall 8), plus the trace for the write-up
  * split_cohort()/holdout_report(): practical rule 2's instruments
  * fallback_eps_F(): App. H arithmetic, exact
  * certificate persistence: save/load round-trip, missing-domain None,
    and the model-version staleness warning

No online sanity test accompanies this chapter ON PURPOSE (same reasoning as
ch. 12): risk_control.py has no LLM/search seam that could gracefully degrade
-- it is arithmetic plus file I/O. run_pipeline/truth are injected seams whose
real (live) implementations arrive with the Sec. 15 harness; the fakes here
pin their CONTRACT (resolved records in, TruthEntity out) exactly.
"""
import json
import math
import warnings

import pytest

from webagg import config
from webagg.type_defs import CorroboratedValue
from webagg.risk_control import (TruthRecord, TruthEntity, usd,
                                 default_truth_key, match_to_truth,
                                 fidelity_loss, hoeffding_p, learn_then_test,
                                 split_cohort, holdout_report, fallback_eps_F,
                                 FidelityCertificate, save_fidelity_cert,
                                 load_fidelity_cert, load_fidelity_cert_record)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class FakeRecord:
    """Pin the slice of ResolvedRecord the module reads: .attributes (a dict
    of CorroboratedValue) and .record_kind. Using a fake keeps the tests
    honest about what the contract actually is."""
    def __init__(self, kind="funding_round", **attrs):
        self.record_kind = kind
        self.attributes = {
            k: CorroboratedValue(value=str(v), belief=0.9, nu=1,
                                 component_sizes=[1],
                                 value_num=(float(v) if isinstance(v, (int, float))
                                            else None))
            for k, v in attrs.items()}


def _truth(*amounts_and_keys):
    recs = tuple(TruthRecord(key=k, amount=a) for a, k in amounts_and_keys)
    return TruthEntity(entity_id="acme", records=recs)


# ---------------------------------------------------------------------------
# usd() and key extraction
# ---------------------------------------------------------------------------

def test_usd_prefers_value_num_then_canonicalizer_then_zero():
    assert usd(CorroboratedValue(value="whatever", belief=1, nu=1,
                                 component_sizes=[1], value_num=4e7)) == 4e7
    # no value_num -> parse the raw string through the shared canonicalizer
    assert usd(CorroboratedValue(value="$40M", belief=1, nu=1,
                                 component_sizes=[1])) == 40_000_000.0
    # unreadable -> 0.0 (and if such a record matches truth, the shortfall
    # correctly shows up as fidelity error -- tested below)
    assert usd(CorroboratedValue(value="Series B", belief=1, nu=1,
                                 component_sizes=[1])) == 0.0
    assert usd(None) == 0.0


def test_default_truth_key_priority():
    r = FakeRecord(amount=1.0, registry_key="0001-23-000045", date="2026-01-05")
    assert default_truth_key(r) == "0001-23-000045"      # registry key wins
    r2 = FakeRecord(amount=1.0, date="2026-01-05")
    assert default_truth_key(r2) == "funding_round|2026-01-05"
    assert default_truth_key(FakeRecord(amount=1.0)) is None   # -> spurious


# ---------------------------------------------------------------------------
# match_to_truth(): one-to-one, over-splits punished, missing excluded
# ---------------------------------------------------------------------------

def test_match_is_one_to_one_so_oversplit_duplicate_is_spurious():
    truth = _truth((40e6, "funding_round|2026-01-05"))
    dup = [FakeRecord(amount=40e6, date="2026-01-05"),
           FakeRecord(amount=40e6, date="2026-01-05")]     # ER over-split
    aligned = match_to_truth(dup, truth)
    matched = [t for (_, t) in aligned if t is not None]
    assert len(matched) == 1          # the truth record absorbs at most one
    assert sum(1 for (_, t) in aligned if t is None) == 1  # dup is spurious


def test_missing_truth_record_never_appears_in_alignment():
    truth = _truth((40e6, "k1"), (25e6, "k2"))     # k2 never reassembled
    aligned = match_to_truth([FakeRecord(amount=40e6, registry_key="k1")], truth)
    assert len(aligned) == 1                        # only contributed records


# ---------------------------------------------------------------------------
# fidelity_loss(): the guide's arithmetic, case by case
# ---------------------------------------------------------------------------

def test_perfect_reassembly_scores_zero():
    truth = _truth((40e6, "k1"), (25e6, "k2"))
    recs = [FakeRecord(amount=40e6, registry_key="k1"),
            FakeRecord(amount=25e6, registry_key="k2")]
    assert fidelity_loss(recs, truth) == 0.0


def test_misread_amount_is_relative_error():
    truth = _truth((40e6, "k1"))
    recs = [FakeRecord(amount=44e6, registry_key="k1")]   # read $44M for $40M
    assert fidelity_loss(recs, truth) == pytest.approx(4e6 / 40e6)


def test_spurious_record_charges_full_value():
    # a mis-join pulled a foreign $10M round into a $40M entity
    truth = _truth((40e6, "k1"))
    recs = [FakeRecord(amount=40e6, registry_key="k1"),
            FakeRecord(amount=10e6, registry_key="not_in_truth")]
    assert fidelity_loss(recs, truth) == pytest.approx(10e6 / 40e6)


def test_missing_record_is_completeness_not_fidelity():
    # pipeline contributed only 1 of 2 true records, but read it perfectly:
    # L_g = 0 -- the miss belongs to eps_C, not eps_F (guide 13.2)
    truth = _truth((40e6, "k1"), (25e6, "k2"))
    assert fidelity_loss([FakeRecord(amount=40e6, registry_key="k1")],
                         truth) == 0.0


def test_unreadable_amount_on_matched_record_is_fidelity_error():
    truth = _truth((40e6, "k1"))
    r = FakeRecord(registry_key="k1")
    r.attributes["amount"] = CorroboratedValue(value="Series B", belief=1,
                                               nu=1, component_sizes=[1])
    assert fidelity_loss([r], truth) == pytest.approx(1.0)   # 40e6/40e6, capped


def test_loss_caps_at_one():
    truth = _truth((1e6, "k1"))
    recs = [FakeRecord(amount=99e6, registry_key="spurious")]
    assert fidelity_loss(recs, truth) == 1.0


# ---------------------------------------------------------------------------
# hoeffding_p()
# ---------------------------------------------------------------------------

def test_hoeffding_exact_value_and_short_circuit():
    # mean 0.05, eps 0.10, n=100 -> exp(-2*100*0.05^2) = exp(-0.5)
    losses = [0.05] * 100
    assert hoeffding_p(losses, 0.10) == pytest.approx(math.exp(-0.5))
    # sample mean at/above eps_F is no evidence at all
    assert hoeffding_p([0.10] * 100, 0.10) == 1.0
    assert hoeffding_p([0.50] * 100, 0.10) == 1.0


def test_hoeffding_more_entities_more_evidence():
    assert hoeffding_p([0.05] * 200, 0.10) < hoeffding_p([0.05] * 50, 0.10)


# ---------------------------------------------------------------------------
# learn_then_test(): the fixed-sequence discipline
# ---------------------------------------------------------------------------

def _ltt_harness(loss_by_config):
    """Fake run_pipeline/truth pinning the injected-seam CONTRACT.

    Each entity has one $40M true record; the fake pipeline returns a record
    whose amount is off by exactly loss_by_config[lam_name] * 40M, so
    fidelity_loss reproduces the loss we scripted -- the fake exercises the
    REAL loss path rather than stubbing it out.
    """
    # 40 entities: Hoeffding needs n -- with only 4, even a loss of 0.01
    # cannot be certified at eps_F=0.30 (p ~ 0.51). A cohort too small to
    # certify anything is a real failure mode worth remembering.
    truth_tbl = {f"g{i}": _truth((40e6, "k1")) for i in range(40)}
    calls = []

    def run_pipeline(g, lam):
        calls.append(lam["name"])
        L = loss_by_config[lam["name"]]
        return [FakeRecord(amount=40e6 * (1 + L), registry_key="k1")]

    return truth_tbl, calls, run_pipeline


def test_ltt_certifies_prefix_and_stops_at_first_failure():
    # cheap passes, mid passes, bad fails, never fails harder -> must stop
    # at 'bad' and NEVER evaluate 'never' (practical rule 1 / pitfall 8)
    truth_tbl, calls, run = _ltt_harness(
        {"cheap": 0.01, "mid": 0.02, "bad": 0.50, "never": 0.0})
    cohort = list(truth_tbl)
    configs = [{"name": n} for n in ("cheap", "mid", "bad", "never")]
    trace = []
    out = learn_then_test(cohort, configs, eps_F=0.30, delta_F=0.20,
                          run_pipeline=run, truth=truth_tbl.get, trace=trace)
    assert out is not None
    lam, mean = out
    assert lam["name"] == "mid"                  # deepest certified config
    assert mean == pytest.approx(0.02)
    assert "never" not in calls                  # fixed-sequence stop held
    assert [t["passed"] for t in trace] == [True, True, False]


def test_ltt_returns_none_when_first_config_fails():
    truth_tbl, _, run = _ltt_harness({"only": 0.9})
    out = learn_then_test(list(truth_tbl), [{"name": "only"}],
                          eps_F=0.10, delta_F=0.05,
                          run_pipeline=run, truth=truth_tbl.get)
    assert out is None


# ---------------------------------------------------------------------------
# practical rule 2: split + holdout report
# ---------------------------------------------------------------------------

def test_split_is_deterministic_and_disjoint():
    cohort = [f"g{i}" for i in range(10)]
    cal1, val1 = split_cohort(cohort, seed=7)
    cal2, val2 = split_cohort(cohort, seed=7)
    assert (cal1, val1) == (cal2, val2)          # reproducible
    assert set(cal1).isdisjoint(val1)
    assert sorted(cal1 + val1) == sorted(cohort)


def test_holdout_report_fields():
    rep = holdout_report([0.02, 0.04, 0.20], eps_F=0.10)
    assert rep["n"] == 3
    assert rep["mean_loss"] == pytest.approx(0.26 / 3)
    assert rep["rate_within"] == pytest.approx(2 / 3)
    assert 0.0 < rep["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# App. H fallback
# ---------------------------------------------------------------------------

def test_fallback_arithmetic_exact():
    # rho + dE + dB*vs/S + a*vm/S + dC + dT
    got = fallback_eps_F(rho_phi=0.02, delta_E=0.05, delta_B=0.03,
                         v_split=10e6, v_merge=5e6, alpha=0.04,
                         delta_C=0.05, delta_T=0.01, total_sum=100e6)
    assert got == pytest.approx(0.02 + 0.05 + 0.03 * 0.1 + 0.04 * 0.05
                                + 0.05 + 0.01)


# ---------------------------------------------------------------------------
# certificate persistence
# ---------------------------------------------------------------------------

def test_cert_roundtrip_missing_none_and_model_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FIDELITY_CERT_DIR", tmp_path)
    assert load_fidelity_cert("nowhere") is None      # never calibrated

    cert = FidelityCertificate(domain="startup_funding", eps_F=0.08,
                               delta_F=0.05, method="ltt",
                               lam={"tau_plus": 0.85}, mean_loss=0.03, n_cal=40)
    path = save_fidelity_cert(cert)
    assert json.loads(open(path).read())["eps_F"] == 0.08

    # the number the Sec. 14 report layer reads
    assert load_fidelity_cert("startup_funding") == 0.08
    back = load_fidelity_cert_record("startup_funding")
    assert back.lam == {"tau_plus": 0.85} and back.created  # stamped on save

    # model drift -> warning, but the cert is still returned (guide 13.4)
    monkeypatch.setattr(config, "MODEL_STRONG", "some-future-model")
    with pytest.warns(UserWarning, match="recalibrate"):
        assert load_fidelity_cert("startup_funding") == 0.08
