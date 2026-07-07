from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .type_defs import Source, Mention


class SourceClass(str, Enum):
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
    """Map a Source to its class via domain lookup (Def. 14's chi)."""
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
# 2. Attribute-coverage matrix (design doc Definition 15)
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Per-record fragmentation analysis result.

    U       = coverage union: attrs contributed by ANY class   (Def. 15)
    K[C]    = per-class coverage: attrs class C contributed    (Def. 15)
    matrix  = M(rho)[a, C] in {0, 1}                           (Def. 15)
    case    = which of Def. 16's three cases this record is in
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
    """Build M(rho) for one resolved record (Def. 15).

    record_mentions: all Mentions grouped into this record by ER.
    query_attributes: the attribute set A the QUERY asks for -- a parameter,
    never hardcoded (impl guide Sec. 10.10: attribute-name drift pitfall).
    """
    K: dict[SourceClass, set[str]] = defaultdict(set)
    matrix: dict[tuple[str, SourceClass], int] = {}
    for m in record_mentions:
        if m.attribute not in query_attributes:
            continue  # off-query attributes don't enter the routing decision
        klass = classify(source_lookup[m.source_id])
        K[klass].add(m.attribute)
        matrix[(m.attribute, klass)] = 1
    # U = union over all classes of what each contributed (Def. 15)
    U: set[str] = set().union(*K.values()) if K else set()
    return CoverageReport(record_id=record_id, U=U, K=dict(K), matrix=matrix)


# ---------------------------------------------------------------------------
# 3. The three-case classifier (design doc Definition 16 / Algorithm 3)
# ---------------------------------------------------------------------------

def classify_fragmentation(report: CoverageReport) -> CoverageReport:
    """Decide scan / join / redundant for one record. Mutates + returns report."""
    if not report.U:
        # Extraction produced nothing for the query attrs -- "empty" case,
        # flagged separately per Def. 16 (missing-attribute, not fragmentation)
        report.case = "empty"
        return report

    # Case 1 (scan-sufficient)
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

    # Case 2 (complementarily fragmented)
    single_class_attrs = {a for a, cs in cls_per_attr.items() if len(cs) == 1}
    if single_class_attrs:
        report.case = "fragmented"
        report.fragmenting_attrs = single_class_attrs
        return report

    # Case 3 (redundant)
    report.case = "redundant"
    return report


# ---------------------------------------------------------------------------
# 4. Single-class sufficiency -> frontier pruning (design doc Definition 17)
# ---------------------------------------------------------------------------

def single_class_sufficiency(reports: list[CoverageReport],
                             gamma: float = 0.90
                             ) -> tuple[bool, SourceClass | None]:
    """Should the agent stop searching other classes entirely?

    Fires when >= gamma of observed records are scan-sufficient under the
    SAME class (Def. 17). Uses only data already collected -- the "free
    signal" of design doc Remark 4; zero extra fetches or LLM calls.
    """
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

    Used only by the prune (Def. 17); a wrong guess costs fetches, never
    correctness (Corollary 3). Deliberately NOT an LLM call.
    """
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


# ---------------------------------------------------------------------------
# 5. Cross-entity contamination guard (design doc Section 6.9)
# ---------------------------------------------------------------------------

def entity_mentioned(source: Source, entity_surfaces: list[str]) -> bool:
    """Does the source's text actually NAME the entity?

    A fragmenting attribute rides on ER alone (single class, no
    corroboration), so we demand the asserting page literally names the
    entity; otherwise it may be Theorem 4's false-merge channel leaking in.
    Surfaces shorter than 4 chars are skipped (too many false positives).
    """
    text = (source.main_text or "").lower()
    return any(s.lower() in text for s in entity_surfaces if len(s) >= 4)


def guarded_fragmenting_value(mention: Mention, source: Source,
                              entity_surfaces: list[str],
                              default_belief: float) -> tuple[str, float]:
    """Return (value, belief); belief halved when the entity isn't named.

    Converts a fragmenting attribute into a probabilistic one with belief
    discounted by entity-mention strength (design doc Sec. 6.9) -- keeps
    Theorem 4's false-merge channel honest at the record level.
    """
    if entity_mentioned(source, entity_surfaces):
        return mention.value, default_belief
    return mention.value, default_belief * 0.5


# ---------------------------------------------------------------------------
# 6. Pipeline wiring helper (impl guide Sec. 10.7)
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
    one frag_case measurement per record (impl guide Sec. 10.8).
    Logging is optional so offline notebooks can call this without a DB.
    """
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
