import pytest

from webaggs.conformal import ConformalError, ConformalGate
from webaggs.er import resolve_entity
from webaggs.fetch import classify_source, domain_of, fetch_url, html_to_text


# --- deny list: the withheld oracle ---------------------------------------
def test_oracle_domains_are_denied_without_any_network_call():
    for url in ("https://api.openalex.org/works?filter=cites:W2626778328",
                "https://openalex.org/W2626778328"):
        src = fetch_url(url, "fm1")       # would raise if it tried the network
        assert src.status == "denied" and src.main_text == ""


def test_scholar_aggregators_are_NOT_denied():
    """Semantic Scholar etc. play the hockey-reference role: part of 'the
    web', only the oracle is withheld. (No network: just the deny check.)"""
    from webaggs.fetch import is_denied
    assert not is_denied("https://www.semanticscholar.org/paper/x")
    assert not is_denied("https://arxiv.org/abs/1706.03762")


def test_domain_and_class():
    assert domain_of("https://www.semanticscholar.org/x") == "semanticscholar.org"
    assert classify_source("https://someblog.net/x") == "other"


def test_html_to_text_strips_script_and_keeps_table_cells():
    html = ("<html><head><script>evil()</script></head><body>"
            "<table><tr><td>10.1234/aaa.1</td><td>2019</td></tr></table></body></html>")
    text = html_to_text(html)
    assert "10.1234/aaa.1" in text and "evil" not in text


# --- ER: target-paper guard ------------------------------------------------
def test_resolve_entity_folds_title_variants_and_rejects_other_papers():
    t = "Attention Is All You Need"
    assert resolve_entity("Attention Is All You Need", t)
    assert resolve_entity("attention is all you need (NIPS 2017)", t)
    assert resolve_entity("ATTENTION IS ALL YOU NEED.", t)
    assert resolve_entity("Some Other Famous Paper", t) is None
    assert resolve_entity("", t) is None
    # two-word containment is too collision-prone: rejected
    assert resolve_entity("Attention Is", t) is None


# --- conformal gate (unchanged machinery) -----------------------------------
def test_gate_threshold_is_split_conformal_quantile():
    pairs = [(c / 100, True) for c in range(70, 100)]
    gate = ConformalGate.fit(pairs, delta_e=0.10)
    assert abs(gate.threshold - 0.28) < 1e-9
    assert gate.accepts(0.75) and not gate.accepts(0.60)


def test_gate_refuses_cold_start_and_tiny_sets():
    with pytest.raises(ConformalError):
        ConformalGate.fit([(0.9, True)] * 5)
    with pytest.raises(ConformalError):
        ConformalGate.load("/nonexistent/calibration.csv")
