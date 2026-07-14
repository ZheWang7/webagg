"""Smoke test for guide ch. 6 (the four-stage reader gate).

Run:  pytest tests/test_smoke_ch6.py -v
All LLM calls are mocked -- no API keys needed. One test per edit:
  validators.py    -> confusion-suite tests
  calibration.py   -> gate threshold math + accept/abstain + bootstrap
  extract.py       -> dual extraction agreement/disagreement, claims
  audit.py         -> stratified sampling + Clopper-Pearson bound
  pipeline wiring  -> RejectedSourceRow is exercised via audit tests
"""
import json
from datetime import datetime

import pytest
from scipy.stats import beta as beta_dist

from webagg.type_defs import Mention, Source
from webagg.validators import (Reject, ExtractionContext, parse_amount,
                               canonicalize_money, validate_mention,
                               stage_plausible)
from webagg.calibration import ConformalGate, nonconf
from webagg.storage import get_session, RejectedSourceRow

T0 = datetime(2026, 1, 15, 12, 0)   # project convention: UTC-naive


def mk_mention(value="EUR 37M", attribute="amount", currency="EUR",
               passage="Acme raised EUR 37M in its Series B.", **kw):
    d = dict(mention_id="s1:amount:aaaa:A", source_id="s1",
             entity_surface="Acme, Inc.", record_kind="funding_round",
             attribute=attribute, value=value, passage=passage,
             extracted_at=T0, currency=currency, self_conf=0.9,
             extractor_id="A")
    d.update(kw)
    return Mention(**d)


# ------------------------------------------------------------- validators.py
def test_parse_amount_unit_grammar():
    assert parse_amount("$40MM") == (40.0, "mm")      # MM is millions, not 40
    assert parse_amount("0.04B") == (0.04, "b")       # 0.04 billion = 40M
    assert parse_amount("40,000,000") == (40000000.0, "")
    with pytest.raises(Reject):
        parse_amount("Series B")                       # not money at all


def test_currency_required_and_fx_flagged():
    with pytest.raises(Reject) as e:
        canonicalize_money("$40M", None, T0)           # money MUST carry currency
    assert e.value.reason == "currency_missing"
    usd, flags = canonicalize_money("EUR 37M", "EUR", T0)
    assert usd == pytest.approx(37e6 * 1.08)           # static FX (provisional)
    assert "fx:EUR" in flags and "fx_static_rate" in flags
    usd, flags = canonicalize_money("$40M", "USD", T0)
    assert usd == 40e6 and flags == []                 # USD: no conversion


def test_cumulative_is_fatal():
    m = mk_mention(value="$63M", currency="USD",
                   passage="bringing its total raised to date to $63M")
    with pytest.raises(Reject) as e:
        validate_mention(m, ExtractionContext())
    assert e.value.reason == "cumulative_not_round"


def test_round_exceeding_cumulative_is_fatal():
    m = mk_mention(value="$80M", currency="USD")
    with pytest.raises(Reject) as e:
        validate_mention(m, ExtractionContext(cumulative=63e6))
    assert e.value.reason == "round_exceeds_cumulative"


def test_flags_extension_postmoney_magnitude():
    m = mk_mention(value="EUR 280M", currency="EUR",
                   passage="Acme extended its Series B, a top-up of EUR 280M")
    m = validate_mention(m, ExtractionContext(stage="series a",
                                              post_money=300e6))
    assert "series_extension" in m.validator_flags
    assert "amount_near_postmoney" in m.validator_flags   # 302M > 0.8 * 300M
    assert "magnitude_outlier" in m.validator_flags       # 302M >> series-a band
    assert m.value_num == pytest.approx(280e6 * 1.08)


def test_date_role_flag_and_stage_bands():
    m = mk_mention(value="2026-01-10", attribute="date", currency=None,
                   passage="announced on Jan 10", date_role=None)
    assert "date_role_missing" in validate_mention(m).validator_flags
    assert stage_plausible("series b", 40e6)
    assert not stage_plausible("seed", 40e6)
    assert stage_plausible(None, 40e6)                     # nothing to check


# ------------------------------------------------------------ calibration.py
def test_threshold_index_math():
    # n=99, delta_E=0.05: t_hat is the ceil(0.95*100)=95th smallest score.
    gate = ConformalGate(delta_E=0.05)
    gate.fit([(str(i), str(i), 1.0 - i / 100.0) for i in range(1, 100)])
    # scores are i/100 for i=1..99 (all correct: 1 - self_conf)
    assert gate.threshold() == pytest.approx(0.95)

    # wrong predictions score >= 1 and sort to the top
    assert nonconf("40000000", "40000000", 0.9) == pytest.approx(0.1)
    assert nonconf("80000000", "40000000", 0.9) == pytest.approx(2.0)  # 100% off
    assert nonconf("oops", "40000000", 0.9) == 2.0


def test_gate_accept_abstain_and_bootstrap():
    gate = ConformalGate(delta_E=0.05)
    gate.fit([("v", "v", 0.8)] * 99)                # all scores = 0.2
    confident = mk_mention(self_conf=0.9)           # score 0.1 <= 0.2
    hesitant = mk_mention(self_conf=0.5)            # score 0.5 >  0.2
    assert gate.accept(confident) is True
    assert gate.accept(hesitant) is False           # abstains: cost, not error

    boot = ConformalGate()                          # UNFITTED: bootstrap mode
    m = mk_mention(self_conf=0.1)
    assert boot.accept(m) is True                   # accept-all, but auditable:
    assert "gate_uncalibrated" in m.validator_flags


# ----------------------------------------------------------------- extract.py
def _payload_a():
    return {"mentions": [
        {"entity_surface": "Acme, Inc.", "record_kind": "funding_round",
         "attribute": "amount", "value": "$40M", "currency": "USD",
         "date_role": None, "t_asof": "2026-01-10", "self_conf": 0.9,
         "passage": "Acme raised $40M in its Series B."},
        {"entity_surface": "Bolt Ltd.", "record_kind": "funding_round",
         "attribute": "amount", "value": "$15M", "currency": "USD",
         "date_role": None, "t_asof": None, "self_conf": 0.8,
         "passage": "Bolt raised $15M."}],
        "claims": [
        {"stratum_surface": "Acme, Inc.", "functional": "SUM",
         "attribute": "amount", "value_num": 63e6, "currency": "USD",
         "t_asof": None, "scope": "to date", "tolerance": 5e5,
         "passage": "bringing its total raised to date to $63M"}]}


def _payload_b():
    return {"mentions": [
        {"entity_surface": "Acme Inc",           # same entity, different surface
         "record_kind": "funding_round", "attribute": "amount",
         "value": "40000000",                    # same value, different spelling
         "currency": "USD", "date_role": None, "t_asof": None,
         "self_conf": 0.85, "passage": "a $40M Series B"},
        {"entity_surface": "Bolt Ltd.", "record_kind": "funding_round",
         "attribute": "amount", "value": "$50M",  # DISAGREES with A's $15M
         "currency": "USD", "date_role": None, "t_asof": None,
         "self_conf": 0.7, "passage": "Bolt raised $50M."}],
        "claims": [
        {"stratum_surface": "Acme, Inc.", "functional": "SUM",
         "attribute": "amount", "value_num": 63.2e6, "currency": "USD",
         "t_asof": None, "scope": "to date", "tolerance": 5e5,
         "passage": "roughly $63M raised to date"}]}


def test_dual_extraction_agreement_and_abstention(monkeypatch):
    from webagg import extract
    payloads = {"extraction_A": _payload_a(), "extraction_B": _payload_b()}
    monkeypatch.setattr(extract, "call_llm",
                        lambda **kw: {"payload": payloads[kw["purpose"]]})
    src = Source(source_id="s1", url="https://reuters.com/x", domain="reuters.com",
                 fetch_time=T0, publish_time=T0, title="t",
                 main_text="...", formulation_id="f1")

    mentions, claims, info = extract.extract_certified(src, "acme funding")
    # Acme $40M: A and B agree AFTER canonicalization ("$40M" == "40000000")
    assert len(mentions) == 1 and mentions[0].entity_surface == "Acme, Inc."
    assert mentions[0].accepted is True
    assert mentions[0].self_conf == pytest.approx(0.85)   # min of A, B
    assert mentions[0].value_num == 40e6                  # validators canonicalized
    assert mentions[0].extractor_id == "A"
    # Bolt: A says $15M, B says $50M -> disagreement -> abstain (no Bolt mention)
    assert info["disagreed"] == 1 and info["agreed"] == 1
    # Claims: SUM 63.0M vs 63.2M within tolerance(5e5) -> kept
    assert len(claims) == 1 and claims[0].functional == "SUM"
    # bi-temporal default: t_asof parsed for Acme's mention
    assert mentions[0].t_asof == datetime(2026, 1, 10)


def test_gate_wired_into_extract_certified(monkeypatch):
    from webagg import extract
    payloads = {"extraction_A": _payload_a(), "extraction_B": _payload_b()}
    monkeypatch.setattr(extract, "call_llm",
                        lambda **kw: {"payload": payloads[kw["purpose"]]})
    src = Source(source_id="s1", url="https://reuters.com/x", domain="reuters.com",
                 fetch_time=T0, publish_time=T0, title="t",
                 main_text="...", formulation_id="f1")
    strict = ConformalGate(delta_E=0.05)
    strict.fit([("v", "v", 0.99)] * 99)          # threshold 0.01: nothing passes
    mentions, _, info = extract.extract_certified(src, "q", gate=strict)
    assert mentions == [] and info["gate_abstains"] == 1


# ------------------------------------------------------------------ audit.py
def test_phi_fn_upper_clopper_pearson(tmp_path, monkeypatch):
    from webagg import audit
    session = get_session(str(tmp_path / "ch6.sqlite"))
    # 60 rejections; exactly one (score .49, near-threshold) was a mistake.
    for i in range(60):
        session.add(RejectedSourceRow(
            source_id=f"r{i}", url=f"https://x.com/{i}",
            rejection_score=i / 120.0, main_text=f"doc {i}"))
    session.commit()
    monkeypatch.setattr(audit, "adjudicate_relevance",
                        lambda text, q: text == "doc 58")   # 1 false negative

    rho = audit.phi_fn_upper(session, "acme funding", n_audit=60, seed=1)
    assert rho == pytest.approx(float(beta_dist.ppf(0.95, 2, 59)))  # ~0.078
    # the audit is cumulative: verdicts stored, rows marked audited
    audited = session.query(RejectedSourceRow).filter_by(audited=True).count()
    assert audited == 60
    assert session.query(RejectedSourceRow).filter_by(audit_verdict=True).count() == 1


def test_phi_fn_upper_zero_findings_not_zero_bound(tmp_path, monkeypatch):
    """0-in-n is NOT proof of a zero FN rate (finite-sample honesty)."""
    from webagg import audit
    session = get_session(str(tmp_path / "ch6b.sqlite"))
    for i in range(60):
        session.add(RejectedSourceRow(source_id=f"r{i}", url="u",
                                      rejection_score=0.1, main_text="doc"))
    session.commit()
    monkeypatch.setattr(audit, "adjudicate_relevance", lambda text, q: False)
    rho = audit.phi_fn_upper(session, "q", n_audit=60, seed=1)
    assert 0.0 < rho == pytest.approx(float(beta_dist.ppf(0.95, 1, 60)))  # ~0.049
