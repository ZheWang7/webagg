"""Ch. 12 offline tests: fragmentation -- join or scan? (guide §12, paper App. D)

One test per duty, so a failure points at the responsible piece:
  * the rule-based source-class classifier (never an LLM call),
  * the coverage matrix + three cases of Def. 14 -- pinned to the paper's
    own worked Example 8 (Acme's Series B),
  * agreement of the two CoverageReport builders (post-ER mentions vs the
    mid-loop class-set accumulator),
  * the gamma = 0.9 single-class sufficiency prune (Cor. 2),
  * the cross-entity contamination guard (0.5x + "weak_entity_link"),
  * the §12 ACCEPTANCE check: misclassifying a scan-sufficient record as
    fragmented leaves the answer unchanged -- only cost moves.

No online sanity test accompanies this chapter ON PURPOSE: §12 has no
LLM/search seam that could gracefully degrade (the classifier is a domain
lookup, the routing is set algebra), so there is no degraded output for a
live check to refuse.
"""
from datetime import datetime

from webagg.type_defs import Source, Mention, CorroboratedValue
from webagg.fragmentation import (SourceClass, classify, coverage_matrix,
                                  coverage_from_class_sets,
                                  classify_fragmentation,
                                  single_class_sufficiency,
                                  maybe_prune_single_class,
                                  keyword_class_predictor,
                                  contamination_guard, entity_mentioned)
from webagg.frontier import FrontierState, Formulation
from webagg.corroboration import corroborate, QTable

NOW = datetime(2026, 7, 1, 12, 0, 0)     # UTC-naive by repo convention


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _src(sid: str, domain: str, text: str = "") -> Source:
    """Minimal Source; classify() only reads .domain, the guard reads
    .main_text."""
    return Source(source_id=sid, url=f"https://{domain}/x", domain=domain,
                  fetch_time=NOW, publish_time=None, title=None,
                  main_text=text, formulation_id="f0")


def _mention(mid: str, sid: str, attr: str, value: str,
             surface: str = "Acme Robotics") -> Mention:
    return Mention(mention_id=mid, source_id=sid, entity_surface=surface,
                   record_kind="funding_round", attribute=attr, value=value,
                   passage=f"{surface} {attr} {value}", extracted_at=NOW)


# ---------------------------------------------------------------------------
# 1. rule-based classifier
# ---------------------------------------------------------------------------

def test_classifier_domain_rules():
    cases = {
        "sec.gov": SourceClass.REGULATORY,
        "www.sec.gov": SourceClass.REGULATORY,          # www. is stripped
        "efts.sec.gov": SourceClass.REGULATORY,         # subdomain matches
        "crunchbase.com": SourceClass.VENDOR,
        "blog.techcrunch.com": SourceClass.NEWS,
        "a16z.com": SourceClass.INVESTOR,
        "foo-ventures.com": SourceClass.INVESTOR,       # self-hosted VC heuristic
        "linkedin.com": SourceClass.SOCIAL,
        "random-startup-wiki.io": SourceClass.OTHER,    # long tail
    }
    for domain, expected in cases.items():
        assert classify(_src("s", domain)) == expected, domain


# ---------------------------------------------------------------------------
# 2 + 3. Def. 14 on the paper's own worked example (App. D, Example 8)
# ---------------------------------------------------------------------------

# Acme's Series B contribution table, verbatim from the paper:
#   amount, date  -> reg, vendor, news
#   lead          -> vendor, news, investor
#   post_money    -> vendor only
#   employees     -> social only
_ACME = {
    "amount":     {"regulatory", "vendor", "news"},
    "date":       {"regulatory", "vendor", "news"},
    "lead":       {"vendor", "news", "investor"},
    "post_money": {"vendor"},
    "employees":  {"social"},
}


def test_example_8_all_five_attrs_is_fragmented():
    rep = classify_fragmentation(coverage_from_class_sets("acme/round", _ACME))
    # the largest class (vendor) covers four of five, missing employees ->
    # complementarily fragmented, and exactly the two single-class attrs
    # must be joined in (they get the contamination guard downstream)
    assert rep.case == "fragmented"
    assert rep.fragmenting_attrs == {"post_money", "employees"}


def test_example_8_three_attrs_is_scan_sufficient_under_vendor():
    sub = {a: _ACME[a] for a in ("amount", "date", "lead")}
    rep = classify_fragmentation(coverage_from_class_sets("acme/round", sub))
    # "Had only amount/date/lead been asked, rho would be scan-sufficient
    # under C_vendor" (Example 8). News also covers all three; the
    # deterministic enum-order tie-break picks vendor.
    assert rep.case == "scan_sufficient"
    assert rep.scan_class == SourceClass.VENDOR


def test_redundant_and_empty_cases():
    # every attribute multi-class but no single class covers U -> redundant
    red = {"amount": {"regulatory", "vendor"}, "date": {"vendor", "news"},
           "lead": {"news", "regulatory"}}
    assert classify_fragmentation(
        coverage_from_class_sets("r", red)).case == "redundant"
    # nothing extracted for the query attrs -> empty (not fragmentation)
    assert classify_fragmentation(
        coverage_from_class_sets("e", {})).case == "empty"


def test_both_builders_agree():
    """The mid-loop class-set builder and the post-ER mention builder must
    label a record identically -- otherwise the prune and the routing would
    disagree about the same evidence."""
    srcs = {"s1": _src("s1", "crunchbase.com"), "s2": _src("s2", "sec.gov")}
    ms = [_mention("m1", "s1", "amount", "$40M"),
          _mention("m2", "s1", "date", "2026-01-15"),
          _mention("m3", "s2", "amount", "$40M")]
    from_mentions = classify_fragmentation(
        coverage_matrix("acme/round", ms, srcs, {"amount", "date"}))
    from_sets = classify_fragmentation(coverage_from_class_sets(
        "acme/round", {"amount": {"vendor", "regulatory"},
                       "date": {"vendor"}}))
    assert (from_mentions.case, from_mentions.scan_class) == \
           (from_sets.case, from_sets.scan_class) == \
           ("scan_sufficient", SourceClass.VENDOR)


# ---------------------------------------------------------------------------
# 4. gamma = 0.9 sufficiency + the frontier prune (cost-only, Cor. 2)
# ---------------------------------------------------------------------------

def _vendor_scan_acc(n_scan: int, n_frag: int) -> dict:
    """frag_acc fixture: n_scan vendor-only records + n_frag mixed ones."""
    acc = {f"scan{i}|round": {"amount": {"vendor"}, "date": {"vendor"}}
           for i in range(n_scan)}
    acc.update({f"frag{i}|round": {"amount": {"vendor"},
                                   "employees": {"social"}}
                for i in range(n_frag)})
    return acc


def _state_with_formulations() -> FrontierState:
    state = FrontierState()
    for fid, q in (("f_v", "acme crunchbase profile"),
                   ("f_r", "acme sec filing form d"),
                   ("f_n", "acme funding techcrunch announce")):
        state.formulations[fid] = Formulation(formulation_id=fid, query=q)
    # an already-issued off-class formulation: spent money, nothing to save
    spent = Formulation(formulation_id="f_spent", query="acme reuters news")
    spent.issued = True
    state.formulations["f_spent"] = spent
    return state


def test_sufficiency_threshold_and_floor():
    reps = [classify_fragmentation(coverage_from_class_sets(rk, ac))
            for rk, ac in _vendor_scan_acc(9, 1).items()]
    assert single_class_sufficiency(reps, gamma=0.9) == (True, SourceClass.VENDOR)
    reps8 = [classify_fragmentation(coverage_from_class_sets(rk, ac))
             for rk, ac in _vendor_scan_acc(8, 2).items()]
    assert single_class_sufficiency(reps8, gamma=0.9) == (False, None)


def test_maybe_prune_fires_and_zeroes_off_class_pending():
    state = _state_with_formulations()
    keep, dropped = maybe_prune_single_class(
        state, _vendor_scan_acc(9, 1), gamma=0.9, min_records=8)
    assert keep == SourceClass.VENDOR and dropped == 2   # f_r and f_n
    # zero-don't-delete: pruned formulations survive with p_success = 0
    assert state.formulations["f_r"].p_success == 0.0
    assert state.formulations["f_n"].p_success == 0.0
    assert state.formulations["f_v"].p_success > 0.0     # keep-class untouched
    assert state.formulations["f_spent"].p_success > 0.0  # issued: skipped


def test_maybe_prune_respects_min_records_floor():
    """Documented repo deviation: no prune before min_records, however
    unanimous the (tiny) sample looks -- 3/3 = 100% must NOT fire."""
    state = _state_with_formulations()
    keep, dropped = maybe_prune_single_class(
        state, _vendor_scan_acc(3, 0), gamma=0.9, min_records=8)
    assert (keep, dropped) == (None, 0)
    assert state.formulations["f_r"].p_success > 0.0     # nothing zeroed


def test_keyword_predictor_never_llm_smoke():
    assert keyword_class_predictor("acme form d edgar") == SourceClass.REGULATORY
    assert keyword_class_predictor("acme linkedin headcount") == SourceClass.SOCIAL
    assert keyword_class_predictor("acme series b history") == SourceClass.OTHER


# ---------------------------------------------------------------------------
# 5. cross-entity contamination guard (guide §12 last bullet)
# ---------------------------------------------------------------------------

def _cv(belief: float = 0.8) -> CorroboratedValue:
    return CorroboratedValue(value="120", belief=belief, nu=1,
                             component_sizes=[1])


def test_guard_fires_discounts_and_flags():
    srcs = {"s1": _src("s1", "linkedin.com",
                       text="The company grew to 120 employees this year.")}
    ms = [_mention("m1", "s1", "employees", "120")]
    cv = _cv(0.8)
    fired = contamination_guard(cv, ms, srcs, ["Acme Robotics", "Acme"])
    assert fired is True
    assert cv.belief == 0.8 * 0.5                        # the 0.5x discount
    assert "weak_entity_link" in cv.validator_flags       # §14 allocator hook
    # firing twice must not duplicate the flag
    contamination_guard(cv, ms, srcs, ["Acme Robotics"])
    assert cv.validator_flags.count("weak_entity_link") == 1


def test_guard_holds_when_any_source_names_entity():
    # ANY-source deviation: the second source names the entity, so the
    # value commits untouched even though the first page never does
    srcs = {"s1": _src("s1", "linkedin.com", text="120 employees."),
            "s2": _src("s2", "linkedin.com",
                       text="Acme Robotics now employs 120 people.")}
    ms = [_mention("m1", "s1", "employees", "120"),
          _mention("m2", "s2", "employees", "120")]
    cv = _cv(0.8)
    assert contamination_guard(cv, ms, srcs, ["Acme Robotics"]) is False
    assert cv.belief == 0.8 and cv.validator_flags == []


def test_entity_mentioned_skips_short_surfaces():
    src = _src("s1", "linkedin.com", text="ACM digital library page")
    # "ACM" (3 chars) must not count as naming the entity "Acme..."
    assert entity_mentioned(src, ["ACM"]) is False


# ---------------------------------------------------------------------------
# 6. THE §12 ACCEPTANCE CHECK: misrouting moves cost, never the answer
# ---------------------------------------------------------------------------

def test_acceptance_misroute_leaves_answer_unchanged():
    """Guide §12: 'misclassifying a scan-sufficient record as fragmented
    must leave the answer unchanged (only cost moves).'

    Same evidence, two routings. Route A treats amount as scan-sufficient
    (no guard). Route B MISCLASSIFIES it as a fragmenting attribute, so the
    contamination guard runs -- but the pages name the entity, so the guard
    holds and the adopted value AND belief are bit-identical. The only
    thing a misroute can change is which searches we would have paid for
    (Cor. 2)."""
    srcs = {"s1": _src("s1", "crunchbase.com",
                       text="Acme Robotics raised $40 million in its Series B."),
            "s2": _src("s2", "techcrunch.com",
                       text="Acme Robotics announced a $40M round today.")}
    by_value = {"$40M": [_mention("m1", "s1", "amount", "$40M"),
                         _mention("m2", "s2", "amount", "$40M")]}
    qt = QTable()

    cv_scan = corroborate(by_value, srcs, qt)             # route A: no guard
    cv_misrouted = corroborate(by_value, srcs, qt)        # route B: guard runs
    fired = contamination_guard(cv_misrouted, by_value["$40M"], srcs,
                                ["Acme Robotics"])

    assert fired is False
    assert cv_misrouted.value == cv_scan.value
    assert cv_misrouted.belief == cv_scan.belief          # answer unchanged
    assert cv_misrouted.validator_flags == []
