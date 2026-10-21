from webaggs.extract import extract_certified, is_relevant
from webaggs.types import Source


def _src(text):
    return Source(source_id="s1", url="https://a.test/x", domain="a.test",
                  fetch_time="t", formulation_id="fm1", status="ok",
                  main_text=text)


def test_relevance_gate(gate):
    ok, _ = is_relevant(_src("Testo Sampleson scored a lot"), "Testo Sampleson")
    assert ok
    ok, score = is_relevant(_src("unrelated page about soup"), "Testo Sampleson")
    assert not ok and score == 0.0


def test_extraction_applies_typed_validators_and_gate(gate):
    def llm(system, user, **kw):
        return {"season_rows": [
            {"player": "Testo Sampleson", "season": "2019-20",
             "points": 450, "confidence": 0.95},            # accepted
            {"player": "Testo Sampleson", "season": "2018-2020",
             "points": 500, "confidence": 0.95},            # span: typed reject
            {"player": "Testo Sampleson", "season": "2020-21",
             "points": 612, "confidence": 0.10},            # low conf: gate reject
        ], "claims": [
            {"kind": "SUM", "value": 1450},                 # kept
            {"kind": "COUNT", "value": 99},                 # typed bound: dropped
        ]}
    ms, claims, n_rej = extract_certified(_src("Testo Sampleson page"),
                                          "Testo Sampleson", gate, llm=llm)
    assert [(m.season, m.value) for m in ms] == [("2019-20", 450)]
    assert n_rej == 2
    assert [(c.kind, c.value) for c in claims] == [("SUM", 1450.0)]
