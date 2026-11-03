import pytest

from webaggs.conformal import ConformalError, ConformalGate
from webaggs.er import resolve_entity
from webaggs.fetch import classify_source, domain_of, fetch_url, html_to_text


# --- deny list: the withheld oracle ---------------------------------------
def test_oracle_domains_are_denied_without_any_network_call():
    for url in ("https://www.nhl.com/player/connor-mcdavid-8478402",
                "https://api-web.nhle.com/v1/player/8478402/landing",
                "https://www.nhle.com/anything"):
        src = fetch_url(url, "fm1")       # would raise if it tried the network
        assert src.status == "denied" and src.main_text == ""


def test_domain_and_class():
    assert domain_of("https://www.hockey-reference.com/x") == \
        "hockey-reference.com"
    assert classify_source("https://www.hockey-reference.com/x") == \
        "stats_vendor"
    assert classify_source("https://www.quanthockey.com/x") == "stats_vendor"
    assert classify_source("https://www.sportsnet.ca/x") == "news"
    assert classify_source("https://someblog.net/x") == "other"


def test_html_to_text_strips_script_and_keeps_table_cells():
    html = ("<html><head><script>evil()</script></head><body>"
            "<table><tr><td>2015-16</td><td>48</td></tr></table></body></html>")
    text = html_to_text(html)
    assert "2015-16" in text and "48" in text and "evil" not in text


# --- ER contamination guard ------------------------------------------------
def test_resolve_entity_folds_variants_and_rejects_other_players():
    assert resolve_entity("C. McDavid", "Connor McDavid") == "connor mcdavid"
    assert resolve_entity("Connor McDavid Jr.", "Connor McDavid") == "connor mcdavid"
    assert resolve_entity("McDavid", "Connor McDavid") == "connor mcdavid"
    # same surname, different first initial: a DIFFERENT player — never merge
    assert resolve_entity("Kris McDavid", "Connor McDavid") is None
    assert resolve_entity("Connor Bedard", "Connor McDavid") is None


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
