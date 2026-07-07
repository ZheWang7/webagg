# tests/test_fragmentation.py
# Offline unit tests for webagg/fragmentation.py (design doc Sec. 6).
# No network, no LLM, no API keys -- run before any live experiment.

from datetime import datetime

from webagg.fragmentation import (
    SourceClass, CoverageReport, classify, coverage_matrix,
    classify_fragmentation, single_class_sufficiency,
    entity_mentioned, guarded_fragmenting_value, keyword_class_predictor,
)
from webagg.frontier import FrontierState, Formulation, prune_for_single_class
from webagg.type_defs import Source, Mention


# ---------- helpers ----------

def make_source(domain: str, text: str = "some text " * 30) -> Source:
    now = datetime(2026, 1, 1)
    return Source(
        source_id=Source.make_id(f"https://{domain}/x", now),
        url=f"https://{domain}/x",
        domain=domain,
        fetch_time=now,
        publish_time=None,
        title=None,
        main_text=text,
        formulation_id="f0",
    )


def make_mention(source: Source, attribute: str, value: str = "v",
                 entity: str = "Acme Corp") -> Mention:
    return Mention(
        mention_id=f"{source.source_id}:{attribute}:{hash(value) & 0xffff:x}",
        source_id=source.source_id,
        entity_surface=entity,
        record_kind="funding_round",
        attribute=attribute,
        value=value,
        passage=f"{entity} {attribute} = {value}",
        extracted_at=datetime(2026, 1, 1),
    )


def make_report(K_dict: dict[str, list[str]]) -> CoverageReport:
    """Shortcut: build a CoverageReport directly from {class_name: attrs}."""
    K = {SourceClass[k]: set(v) for k, v in K_dict.items()}
    U = set().union(*K.values()) if K else set()
    return CoverageReport(record_id="t", U=U, K=K, matrix={})


# ---------- source-class classifier ----------

def test_classify_domain_rules():
    assert classify(make_source("sec.gov")) == SourceClass.REGULATORY
    assert classify(make_source("www.sec.gov")) == SourceClass.REGULATORY  # www strip
    assert classify(make_source("blog.techcrunch.com")) == SourceClass.NEWS  # subdomain
    assert classify(make_source("crunchbase.com")) == SourceClass.VENDOR
    assert classify(make_source("linkedin.com")) == SourceClass.SOCIAL
    assert classify(make_source("acme-ventures.com")) == SourceClass.INVESTOR  # heuristic
    assert classify(make_source("random-blog.example")) == SourceClass.OTHER


# ---------- coverage matrix ----------

def test_coverage_matrix_builds_U_and_K():
    s_vendor = make_source("crunchbase.com")
    s_social = make_source("linkedin.com")
    lookup = {s.source_id: s for s in (s_vendor, s_social)}
    ms = [make_mention(s_vendor, "amount"), make_mention(s_vendor, "date"),
          make_mention(s_social, "employees"),
          make_mention(s_social, "irrelevant_attr")]  # not in query -> ignored
    rep = coverage_matrix("rec1", ms, lookup, {"amount", "date", "employees"})
    assert rep.U == {"amount", "date", "employees"}
    assert rep.K[SourceClass.VENDOR] == {"amount", "date"}
    assert rep.K[SourceClass.SOCIAL] == {"employees"}
    assert rep.matrix[("amount", SourceClass.VENDOR)] == 1
    assert ("irrelevant_attr", SourceClass.SOCIAL) not in rep.matrix


# ---------- three-case classifier (Def. 16) ----------

def test_scan_sufficient_when_vendor_has_everything():
    r = classify_fragmentation(make_report(
        {"VENDOR": ["amount", "date", "lead"], "NEWS": ["amount", "date"]}))
    assert r.case == "scan_sufficient"
    assert r.scan_class == SourceClass.VENDOR


def test_complementary_when_employee_only_in_social():
    # NOTE: guide deviation -- impl guide Sec. 10.9's fixture omits NEWS yet
    # asserts amount is "shared between vendor and news"; as written, amount
    # IS single-class and the guide's own test would fail. We add NEWS so the
    # fixture matches the comment's intent (design doc Sec. 6.7 worked example).
    r = classify_fragmentation(make_report(
        {"VENDOR": ["amount", "date", "lead", "post_money"],
         "NEWS": ["amount", "date", "lead"],
         "SOCIAL": ["employees"]}))
    assert r.case == "fragmented"
    assert r.fragmenting_attrs == {"employees", "post_money"}  # single-class only
    assert "amount" not in r.fragmenting_attrs  # multi-class attr never fragments


def test_redundant_full_overlap_is_scan_sufficient():
    # Both classes cover U entirely -> case 1 fires first (either class works)
    r = classify_fragmentation(make_report(
        {"VENDOR": ["amount", "date"], "NEWS": ["amount", "date"]}))
    assert r.case == "scan_sufficient"


def test_redundant_partial_overlap_no_single_class_attrs():
    # No class covers U, but every attribute has >= 2 contributors -> redundant
    r = classify_fragmentation(make_report(
        {"VENDOR": ["amount", "date"], "NEWS": ["date", "lead"],
         "REGULATORY": ["amount", "lead"]}))
    assert r.case == "redundant"
    assert r.fragmenting_attrs == set()


def test_empty_U_does_not_crash():
    r = classify_fragmentation(make_report({}))
    assert r.case == "empty"


# ---------- single-class sufficiency + prune (Def. 17) ----------

def test_single_class_sufficiency_pruning():
    rs = [classify_fragmentation(make_report(
        {"VENDOR": ["amount", "date", "lead"]})) for _ in range(9)]
    rs.append(classify_fragmentation(make_report(
        {"NEWS": ["amount", "date", "lead"]})))
    prune, keep = single_class_sufficiency(rs, gamma=0.85)
    assert prune is True and keep == SourceClass.VENDOR


def test_single_class_sufficiency_empty_and_below_gamma():
    assert single_class_sufficiency([], gamma=0.9) == (False, None)
    rs = [classify_fragmentation(make_report(
        {"VENDOR": ["a"], "SOCIAL": ["b"]}))]  # fragmented, not scan-sufficient
    assert single_class_sufficiency(rs, gamma=0.9) == (False, None)


def test_prune_zeroes_off_class_formulations():
    st = FrontierState()
    st.formulations["a"] = Formulation("a", "Acme crunchbase profile",
                                       expected_yield=3.0)
    st.formulations["b"] = Formulation("b", "Acme SEC filing 10-K",
                                       expected_yield=2.0)
    st.formulations["c"] = Formulation("c", "Acme press release news",
                                       expected_yield=2.0)
    st.formulations["c"].issued = True  # already spent -> untouched
    dropped = prune_for_single_class(st, SourceClass.VENDOR,
                                     keyword_class_predictor)
    assert dropped == 1                                    # only "b"
    assert st.formulations["b"].residual_yield == 0.0
    assert st.formulations["a"].residual_yield == 3.0      # kept class survives
    # pruned formulation no longer counts toward U_hat's frontier credit
    assert st.active_frontier_size() == 1


# ---------- contamination guard (Sec. 6.9) ----------

def test_entity_mentioned_and_guard():
    named = make_source("linkedin.com", "Acme Corp has 87 employees today.")
    unnamed = make_source("linkedin.com", "The company grew to 87 employees.")
    m = make_mention(named, "employees", "87")
    surfaces = ["Acme Corp", "Acme, Inc."]
    assert entity_mentioned(named, surfaces) is True
    assert entity_mentioned(unnamed, surfaces) is False
    _, b1 = guarded_fragmenting_value(m, named, surfaces, 0.8)
    _, b2 = guarded_fragmenting_value(m, unnamed, surfaces, 0.8)
    assert b1 == 0.8 and b2 == 0.4  # halved on weak entity link
