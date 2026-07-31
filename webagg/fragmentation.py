"""Fragmentation: join or scan? (impl guide §12, SIGMOD paper App. D)

Per resolved record, decide whether one SOURCE CLASS supplies every needed
attribute (scan) or attributes are spread across classes (join). The paper
frames this as an OPTIMIZER rule: the decision changes cost, never the
answer (App. D, Corollary 2) -- the one exception being cross-entity
contamination, which is the false-merge channel of Thm. 7 and is guarded
at record level here (contamination_guard).

Module map, in the order the guide lists the duties:
  1. SourceClass + classify()            rule-based classifier, NEVER an LLM
                                          call; fetch.py stamps the result
                                          onto Source.source_class and the
                                          corroboration priors read it.
  2. coverage_matrix() / CoverageReport  M(rho)[a, C] of paper Def. 14.
  3. classify_fragmentation()            the three cases of Def. 14.
  4. single_class_sufficiency() +        the gamma=0.9 frontier prune of
     maybe_prune_single_class()          Corollary 2 (a fetch saving only).
  5. contamination_guard()               belief 0.5x + "weak_entity_link"
                                          flag when a fragmenting page never
                                          names the entity (guide §12).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .type_defs import Source, Mention, CorroboratedValue


# ---------------------------------------------------------------------------
# 1. Source-class classifier (guide §12 bullet 1; paper App. D preamble)
# ---------------------------------------------------------------------------

class SourceClass(str, Enum):
    """Classes share a content contract and are rule-classified from
    domain/type -- no LLM call (paper App. D). VOCABULARY NOTE (documented
    deviation, carried from ch. 8): the paper says "reg"; this repo says
    "regulatory", and corroboration.CLASS_PRIOR carries alias entries for
    both spellings plus "social"/"blog"."""
    REGULATORY = "regulatory"   # SEC/EDGAR, ClinicalTrials.gov, USPTO
    VENDOR = "vendor"           # Crunchbase, PitchBook, DealRoom
    NEWS = "news"               # press releases, TechCrunch, Reuters
    INVESTOR = "investor"       # VC portfolio + fund pages
    SOCIAL = "social"           # LinkedIn, company About pages
    OTHER = "other"             # long-tail default bucket


# Rule table: domain suffix -> class
_RULES: list[tuple[set[str], SourceClass]] = [
    ({"sec.gov", "data.sec.gov", "clinicaltrials.gov", "uspto.gov",
      "europa.eu"}, SourceClass.REGULATORY),
    ({"crunchbase.com", "pitchbook.com", "dealroom.co", "tracxn.com"},
     SourceClass.VENDOR),
    ({"prnewswire.com", "businesswire.com", "globenewswire.com",
      "techcrunch.com", "reuters.com", "bloomberg.com", "ft.com",
      "wsj.com", "venturebeat.com"}, SourceClass.NEWS),
    ({"a16z.com", "sequoiacap.com", "kleinerperkins.com",
      "indexventures.com", "accel.com", "lightspeed.com"},
     SourceClass.INVESTOR),
    ({"linkedin.com"}, SourceClass.SOCIAL),
]


def classify(source: Source) -> SourceClass:
    """Map a Source to its class via domain lookup.

    Runs at FETCH TIME: fetch.fetch_url() stamps classify(src).value onto
    Source.source_class, and that stamp is what the corroboration priors
    (corroboration.QTable.q) and the coverage builders below read -- later
    layers never re-derive the class (guide §12 bullet 1)."""
    d = source.domain.lower()
    # strip leading "www." so "www.sec.gov" matches the "sec.gov" rule
    # (same canonicalization issue we hit with reliability() priors)
    if d.startswith("www."):
        d = d[4:]
    for domains, klass in _RULES:
        # exact match OR subdomain match (e.g. "blog.techcrunch.com")
        if any(d == x or d.endswith("." + x) for x in domains):
            return klass
    # heuristic: self-hosted VC pages ("foo-ventures.com", "barcapital.com")
    if any(d.endswith(suf) for suf in ("vc.com", "ventures.com", "capital.com")):
        return SourceClass.INVESTOR
    return SourceClass.OTHER


# ---------------------------------------------------------------------------
# 2. Attribute-coverage matrix (paper Definition 14)
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Per-record fragmentation analysis result.

    U       = coverage union: attrs contributed by ANY class      (Def. 14)
    K[C]    = per-class coverage: attrs class C contributed       (Def. 14)
    matrix  = M(rho)[a, C] in {0, 1}                              (Def. 14)
    case    = which of Def. 14's three cases this record is in
    """
    record_id: str
    U: set[str]
    K: dict[SourceClass, set[str]]
    matrix: dict[tuple[str, SourceClass], int]
    case: str = "undecided"     # scan_sufficient | fragmented | redundant | empty
    scan_class: SourceClass | None = None
    fragmenting_attrs: set[str] = field(default_factory=set)


def coverage_matrix(record_id: str,
                    record_mentions: list[Mention],
                    source_lookup: dict[str, Source],
                    query_attributes: set[str]) -> CoverageReport:
    """Build M(rho) for one resolved record (Def. 14) from its Mentions.

    record_mentions: all Mentions grouped into this record by ER.
    query_attributes: the attribute set A the QUERY asks for -- a parameter,
    never hardcoded (attribute-name drift pitfall). The matrix is computed
    from records already collected: zero marginal fetch cost (Cor. 2)."""
    K: dict[SourceClass, set[str]] = defaultdict(set)
    matrix: dict[tuple[str, SourceClass], int] = {}
    for m in record_mentions:
        if m.attribute not in query_attributes:
            continue  # off-query attributes don't enter the routing decision
        klass = classify(source_lookup[m.source_id])
        K[klass].add(m.attribute)
        matrix[(m.attribute, klass)] = 1
    # U = union over all classes of what each contributed (Def. 14)
    U: set[str] = set().union(*K.values()) if K else set()
    return CoverageReport(record_id=record_id, U=U, K=dict(K), matrix=matrix)


def coverage_from_class_sets(record_id: str,
                             attr_classes: dict[str, set[str]]
                             ) -> CoverageReport:
    """Build a CoverageReport from an {attribute -> {class_str, ...}} map.

    The MID-LOOP builder: run_query keeps this map incrementally as
    mentions arrive (the class string comes straight off the Source stamp),
    so the §12 sufficiency check reads no DB rows and issues no fetches --
    exactly Cor. 2's "zero marginal fetch cost". Produces the same report
    shape as coverage_matrix(), so classify_fragmentation() and
    single_class_sufficiency() run unchanged on either builder."""
    K: dict[SourceClass, set[str]] = defaultdict(set)
    matrix: dict[tuple[str, SourceClass], int] = {}
    for a, classes in attr_classes.items():
        for c in classes:
            klass = SourceClass(c)      # stored as plain strings on Source
            K[klass].add(a)
            matrix[(a, klass)] = 1
    return CoverageReport(record_id=record_id, U=set(attr_classes),
                          K=dict(K), matrix=matrix)


# ---------------------------------------------------------------------------
# 3. The three-case classifier (paper Definition 14, second half)
# ---------------------------------------------------------------------------

def classify_fragmentation(report: CoverageReport) -> CoverageReport:
    """Decide scan / join / redundant for one record. Mutates + returns report.

    Def. 14: scan-sufficient if some class C* has K[C*] >= U; complementarily
    fragmented if no class covers U but the union of classes does AND some
    attribute is single-class (it must be JOINED in); redundantly covered
    otherwise (every attribute multi-class -> route through corroboration).
    Under correct ER, all three routes recover U in full (Lemma 2)."""
    if not report.U:
        # Extraction produced nothing for the query attrs -- "empty" case,
        # flagged separately (a missing-attribute problem, not fragmentation)
        report.case = "empty"
        return report

    # Case 1 (scan-sufficient): one class covers everything any class has.
    # Deterministic pick: biggest K first, then enum order as tie-break.
    for klass in sorted(report.K,
                        key=lambda c: (-len(report.K[c]), list(SourceClass).index(c))):
        if report.K[klass] >= report.U:
            report.case = "scan_sufficient"
            report.scan_class = klass
            return report

    # Reaching here: no single class covers U, i.e. every K[C] is a strict
    # subset of U. Count how many classes contribute each attribute.
    cls_per_attr: dict[str, set[SourceClass]] = defaultdict(set)
    for klass, attrs in report.K.items():
        for a in attrs:
            cls_per_attr[a].add(klass)

    # Case 2 (complementarily fragmented): some attribute exists in exactly
    # one class -- assembling U forces a JOIN, and those single-class
    # attributes ride on ER alone, so they get the contamination guard.
    single_class_attrs = {a for a, cs in cls_per_attr.items() if len(cs) == 1}
    if single_class_attrs:
        report.case = "fragmented"
        report.fragmenting_attrs = single_class_attrs
        return report

    # Case 3 (redundant): every attribute is multi-class; cross-class
    # corroboration (§8) settles each one.
    report.case = "redundant"
    return report


# ---------------------------------------------------------------------------
# 4. Single-class sufficiency -> frontier pruning (guide §12; paper Cor. 2)
# ---------------------------------------------------------------------------

def single_class_sufficiency(reports: list[CoverageReport],
                             gamma: float = 0.90
                             ) -> tuple[bool, SourceClass | None]:
    """Should the agent stop searching other classes entirely?

    Fires when >= gamma of observed records are scan-sufficient under the
    SAME class (Cor. 2, gamma = 0.9). Uses only data already collected --
    zero extra fetches or LLM calls. Empty/fragmented/redundant records sit
    in the denominator, which only makes firing HARDER (conservative)."""
    if not reports:
        return False, None
    counts: dict[SourceClass, int] = defaultdict(int)
    for r in reports:
        if r.case == "scan_sufficient" and r.scan_class is not None:
            counts[r.scan_class] += 1
    if not counts:
        return False, None
    best_class, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count / len(reports) >= gamma:
        return True, best_class
    return False, None


def keyword_class_predictor(query: str) -> SourceClass:
    """Cheap keyword guess of which class a frontier formulation targets.

    Used only by the prune; a wrong guess costs fetches, never correctness
    (Cor. 2). Deliberately NOT an LLM call."""
    q = query.lower()
    if any(w in q for w in ("crunchbase", "pitchbook", "dealroom", "vendor")):
        return SourceClass.VENDOR
    if any(w in q for w in ("filing", "10-k", "10k", "8-k", "form d",
                            "sec ", "edgar", "nct")):
        return SourceClass.REGULATORY
    if any(w in q for w in ("press release", "announce", "news",
                            "techcrunch", "reuters")):
        return SourceClass.NEWS
    if any(w in q for w in ("portfolio", "fund page", "investor page")):
        return SourceClass.INVESTOR
    if any(w in q for w in ("linkedin", "employees", "headcount")):
        return SourceClass.SOCIAL
    return SourceClass.OTHER


def maybe_prune_single_class(state, frag_acc: dict[str, dict[str, set[str]]],
                             *, gamma: float, min_records: int
                             ) -> tuple[SourceClass | None, int]:
    """The mid-loop §12 hook: check sufficiency over everything seen so far
    and, if it fires, zero pending formulations aimed at other classes.

    frag_acc: {record_key: {attribute: {class_str, ...}}}, maintained
    incrementally by run_query as mentions arrive. Records here are PRE-ER
    surface|kind keys -- acceptable because the decision is cost-only
    (Cor. 2): mis-grouped records can at worst delay or skip the prune.

    min_records is a DOCUMENTED REPO DEVIATION: the guide states no floor,
    but gamma over one or two records fires trivially (1/1 = 100%). Waiting
    is the conservative direction -- a late prune wastes fetches, never
    touches the answer.

    Returns (keep_class, n_dropped); (None, 0) when nothing fired."""
    if len(frag_acc) < min_records:
        return None, 0
    reps = [classify_fragmentation(coverage_from_class_sets(rk, ac))
            for rk, ac in frag_acc.items()]
    fires, keep = single_class_sufficiency(reps, gamma=gamma)
    if not fires:
        return None, 0
    # lazy import: frontier is a sibling layer; keep this module import-light
    from .frontier import prune_for_single_class
    dropped = prune_for_single_class(state, keep, keyword_class_predictor)
    return keep, dropped


# ---------------------------------------------------------------------------
# 5. Cross-entity contamination guard (guide §12, last bullet)
# ---------------------------------------------------------------------------

def entity_mentioned(source: Source, entity_surfaces: list[str]) -> bool:
    """Does the source's text actually NAME the entity?

    A fragmenting attribute rides on ER alone (single class, no cross-class
    corroboration), so we demand the asserting page literally names the
    entity; otherwise it may be Thm. 7's false-merge channel leaking in --
    the ONLY way fragmentation can err (Lemma 2). Surfaces shorter than
    4 chars are skipped (too many false positives)."""
    text = (source.main_text or "").lower()
    return any(s.lower() in text for s in entity_surfaces if len(s) >= 4)


def contamination_guard(cv: CorroboratedValue,
                        attr_mentions: list[Mention],
                        source_lookup: dict[str, Source],
                        entity_surfaces: list[str]) -> bool:
    """Before a fragmenting attribute commits, verify the contributing page
    names the entity; if none does, discount belief (0.5x) and add
    "weak_entity_link" to validator_flags so the §14 verification allocator
    can see it (guide §12). Returns True iff the guard FIRED.

    Kept deviation (documented at the pre-SIGMOD introduction of this
    guard): the guide's sketch checks only the FIRST mention's source; we
    accept ANY asserting source that names the entity, which is strictly
    less trigger-happy."""
    # pydantic models aren't hashable; dedupe on the id
    src_ids = {m.source_id for m in attr_mentions}
    if any(entity_mentioned(source_lookup[sid], entity_surfaces)
           for sid in src_ids):
        return False                       # entity named somewhere: commit as-is
    cv.belief = cv.belief * 0.5            # the 0.5x discount of guide §12
    if "weak_entity_link" not in cv.validator_flags:
        cv.validator_flags.append("weak_entity_link")
    return True


# ---------------------------------------------------------------------------
# 6. Pipeline wiring helper (per-record classification + logging)
# ---------------------------------------------------------------------------

def classify_all_records(by_record: dict,
                         source_lookup: dict[str, Source],
                         query_attributes: set[str],
                         *, session=None, run_id: str | None = None,
                         step: int = 0) -> list[tuple]:
    """Run coverage_matrix + classify_fragmentation over every resolved record.

    by_record: {(entity_id, record_kind): [Mention, ...]} -- the grouping the
    pipeline builds right after entity resolution. Returns
    [(key, CoverageReport, mentions), ...] and, if a session is given, logs
    one frag_case measurement per record.
    Logging is optional so offline notebooks can call this without a DB."""
    out = []
    for key, ms in by_record.items():
        record_id = f"{key[0]}/{key[1]}"
        rep = classify_fragmentation(
            coverage_matrix(record_id, ms, source_lookup, query_attributes))
        out.append((key, rep, ms))
        if session is not None and run_id is not None:
            from .metrics import log_measurement  # lazy: keep module import-light
            # coverage_density: filled fraction of the |A| x |C| matrix
            density = (len(rep.matrix)
                       / max(len(query_attributes) * len(SourceClass), 1))
            log_measurement(session, run_id, step, "frag_case", 1.0, extra={
                "record_id": rep.record_id,
                "case": rep.case,
                "scan_class": rep.scan_class.value if rep.scan_class else None,
                "n_fragmenting_attrs": len(rep.fragmenting_attrs),
                "coverage_density": density,
                "U": sorted(rep.U),
                "K": {k.value: sorted(v) for k, v in rep.K.items()},
            })
    return out
