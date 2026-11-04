from tests.conftest import TARGET
from webaggs.extract import extract_certified, is_relevant
from webaggs.types import Source


def _src(text):
    return Source(source_id="s1", url="https://a.test/x", domain="a.test",
                  fetch_time="t", formulation_id="fm1", status="ok",
                  main_text=text)


def test_relevance_gate(gate):
    ok, _ = is_relevant(_src(f"papers citing {TARGET} are listed"), TARGET)
    assert ok
    ok, score = is_relevant(_src("a page about soup recipes"), TARGET)
    assert not ok and score < 0.5


def test_extraction_keys_doi_first_applies_gate_and_disables_claims(gate):
    def llm(system, user, **kw):
        return {"citations": [
            {"title": "Good Paper With Doi", "doi": "DOI: 10.1234/ab.1",
             "cited_paper": TARGET, "confidence": 0.95},     # accepted, doi key
            {"title": "Title Only Citing Paper", "doi": "",
             "cited_paper": TARGET, "confidence": 0.92},     # accepted, t: key
            {"title": "Hi", "doi": "",
             "cited_paper": TARGET, "confidence": 0.95},     # unkeyable: reject
            {"title": "Low Confidence Citing Paper", "doi": "",
             "cited_paper": TARGET, "confidence": 0.10},     # gate reject
        ], "claims": [{"kind": "COUNT", "value": 999}]}      # must be IGNORED
    ms, claims, n_rej = extract_certified(_src("x"), TARGET, gate, llm=llm)
    assert [m.season for m in ms] == ["doi:10.1234/ab.1",
                                      "t:title only citing paper"]
    assert all(m.value == 1 for m in ms)          # COUNT domain: value 1
    assert n_rej == 2
    assert claims == []                           # claims OFF (pre-registered)
