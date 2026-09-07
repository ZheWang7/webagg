"""The ER adjudicator must survive malformed LLM payloads (retry once,
then theta = 0.5 band fallback) instead of crashing a whole replay --
same policy the extract layer already pins for malformed fields."""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webagg.entity_resolution as er


@dataclass
class _M:
    entity_surface: str
    source_id: str
    passage: str


@dataclass
class _S:
    domain: str


def _fixture():
    a = _M("Omsom", "s1", "Omsom raised $3M")
    b = _M("Osome", "s2", "Osome raised $24M")
    lookup = {"s1": _S("techcrunch.com"), "s2": _S("osome.com")}
    return a, b, lookup


def _patch_llm(monkeypatch, payloads):
    """Feed adjudicate_llm a scripted sequence of payloads."""
    seq = iter(payloads)

    def fake_call_llm(system=None, user=None, purpose=None):
        return {"payload": next(seq)}

    import webagg.llm
    monkeypatch.setattr(webagg.llm, "call_llm", fake_call_llm)


def test_well_formed_payload_unchanged(monkeypatch):
    _patch_llm(monkeypatch, [{"match": False, "confidence": 0.9}])
    a, b, lookup = _fixture()
    assert abs(er.adjudicate_llm(a, b, lookup) - 0.1) < 1e-9


def test_malformed_then_good_retries_once(monkeypatch):
    _patch_llm(monkeypatch, [{"verdict": "no"},                 # malformed
                             {"match": True, "confidence": 0.8}])
    a, b, lookup = _fixture()
    assert abs(er.adjudicate_llm(a, b, lookup) - 0.8) < 1e-9


def test_malformed_twice_falls_back_to_band(monkeypatch, capsys):
    _patch_llm(monkeypatch, [{"verdict": "no"},
                             {"match": True, "confidence": "high"}])
    a, b, lookup = _fixture()
    assert er.adjudicate_llm(a, b, lookup) == 0.5
    assert "WARNING" in capsys.readouterr().out   # loud, not silent


def test_out_of_range_confidence_is_clipped(monkeypatch):
    _patch_llm(monkeypatch, [{"match": True, "confidence": 1.7}])
    a, b, lookup = _fixture()
    assert er.adjudicate_llm(a, b, lookup) == 1.0
