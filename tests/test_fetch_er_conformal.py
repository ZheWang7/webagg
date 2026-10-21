import pytest

from webaggs.conformal import ConformalError, ConformalGate
from webaggs.er import resolve_entity
from webaggs.fetch import classify_source, domain_of, fetch_url, html_to_text


# --- deny list: the withheld oracle ---------------------------------------
def test_oracle_domains_are_denied_without_any_network_call():
    for url in ("https://www.nba.com/stats/player/1628973",
                "https://stats.nba.com/player/1628973"):
        src = fetch_url(url, "fm1")       # would raise if it tried the network
        assert src.status == "denied" and src.main_text == ""


def test_domain_and_class():
    assert domain_of("https://www.basketball-reference.com/x") == \
        "basketball-reference.com"
    assert classify_source("https://www.basketball-reference.com/x") == \
        "stats_vendor"
    assert classify_source("https://someblog.net/x") == "other"


def test_html_to_text_strips_script_and_keeps_table_cells():
    html = ("<html><head><script>evil()</script></head><body>"
            "<table><tr><td>2019-20</td><td>450</td></tr></table></body></html>")
    text = html_to_text(html)
    assert "2019-20" in text and "450" in text and "evil" not in text


# --- ER contamination guard ------------------------------------------------
def test_resolve_entity_folds_variants_and_rejects_other_players():
    assert resolve_entity("J. Brunson", "Jalen Brunson") == "jalen brunson"
    assert resolve_entity("Jalen Brunson Jr.", "Jalen Brunson") == "jalen brunson"
    assert resolve_entity("Brunson", "Jalen Brunson") == "jalen brunson"
    # same surname, different first initial: a DIFFERENT player — never merge
    assert resolve_entity("Kris Brunson", "Jalen Brunson") is None
    assert resolve_entity("Jalen Johnson", "Jalen Brunson") is None


# --- conformal gate ----------------------------------------------------------
def test_gate_threshold_is_split_conformal_quantile():
    pairs = [(c / 100, True) for c in range(70, 100)]       # 30 correct examples
    gate = ConformalGate.fit(pairs, delta_e=0.10)
    # scores = 1-conf in [0.01..0.30]; k = ceil(31*0.9) = 28 -> 28th smallest
    assert abs(gate.threshold - 0.28) < 1e-9
    assert gate.accepts(0.75) and not gate.accepts(0.60)


def test_gate_refuses_cold_start_and_tiny_sets():
    with pytest.raises(ConformalError):
        ConformalGate.fit([(0.9, True)] * 5)                # too small
    with pytest.raises(ConformalError):
        ConformalGate.load("/nonexistent/calibration.csv")  # unfitted: refuse
