"""Deterministic SEC Form D parser + ground-truth table builder (guide Sec. 15).

This module is the "one parser, two roles" piece of the withheld-registry
oracle:

  ORACLE ROLE (scripts/build_truth.py): run the EDGAR driver OUTSIDE the
    agent with as_oracle=True and `formd_mentions` as the extract_fn. Every
    Form D / D/A primary document is parsed DETERMINISTICALLY -- pure XML
    tag lookups, no LLM anywhere -- into the per-stratum truth table
    (records, amounts, dates, the true SUM and COUNT) that grades the agent.

  AGENT ROLE (optional): the same `formd_mentions` can serve as a
    deterministic extract_fn for a schema-mode agent run over EDGAR, so
    registry reads never depend on the LLM extractor either.

Why determinism is non-negotiable here: the answer key must not be produced
by the same fallible student it will grade (see run_schema_addressable's
as_oracle refusal). An LLM mis-reading a filing would poison both sides of
the comparison; an XML tag lookup cannot mis-read.

WHAT A FORM D SAYS (the 30-second domain primer): a company selling
securities in an exempt offering (a private funding round) must file a
Form D with the SEC. The primary document is a small XML file. The fields
we rely on:

    offeringData/offeringSalesAmounts/totalAmountSold   -- dollars ACTUALLY
        raised so far in this offering (our ground-truth amount);
    offeringData/typeOfFiling/dateOfFirstSale/value     -- when the round's
        first sale happened (the date the open web tends to report);
    typeOfFiling/newOrAmendment/isAmendment + previousAccessionNumber
        -- a "Form D/A" AMENDS an earlier filing; the chain of amendments
        describes ONE round, and only the newest filing's numbers count.

So the truth table is built in two steps: parse every filing, then COLLAPSE
each amendment chain to its newest filing -- one TruthRecord per round.
This mirrors exactly what the agent's supersession machinery must achieve
from the open web (a D/A's value kills the D's echoes), which is the point:
the oracle computes the right answer by construction, the agent has to earn
it.

Truth-key convention: TruthRecord.key = f"funding_round|{date_of_first_sale}"
(ISO date). This matches risk_control.default_truth_key's fallback exactly
-- the agent side keys a resolved record as f"{record_kind}|{date.value}",
and ISO dates pass through canonicalize_value untouched -- so oracle and
agent produce byte-identical keys without sharing any code path at grading
time. The EDGAR accession number is kept in the per-record metadata for the
stronger "registry_key" match when a source quotes it.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .risk_control import TruthEntity, TruthRecord
from .type_defs import Mention, Source

# Accession numbers look like 0001181412-22-000010 (dashes optional).
_ACCESSION_RX = re.compile(r"\d{10}-?\d{2}-?\d{6}")


# ===========================================================================
# 15.1  One parsed filing
# ===========================================================================

@dataclass(frozen=True)
class FormDFiling:
    """Everything we take from ONE Form D / D/A primary document.

    `amount_sold` is None when the XML carries no readable number (rare;
    the builder then scores it 0 and flags it, never guesses). `indefinite`
    marks an "Indefinite" totalOfferingAmount -- informational only, since
    ground truth uses the SOLD amount, which is always a number.
    """
    accession: str                      # this filing's own accession number
    cik: str                            # 10-digit zero-padded filer id
    entity_name: str                    # issuer's legal name as filed
    is_amendment: bool                  # True for Form D/A
    previous_accession: Optional[str]   # the filing this one amends (D/A only)
    filing_date: Optional[str]          # ISO date the SEC received it
    date_of_first_sale: Optional[str]   # ISO date; None if "yet to occur"
    amount_sold: Optional[float]        # totalAmountSold in USD
    offering_amount: Optional[float]    # totalOfferingAmount in USD (or None)
    indefinite: bool = False            # totalOfferingAmount == "Indefinite"
    parse_flags: tuple[str, ...] = ()   # anything odd, recorded loudly


def _text(root: ET.Element, path: str) -> Optional[str]:
    """findtext + strip, mapping empty/missing to None (one honest 'absent')."""
    s = root.findtext(path)
    s = s.strip() if s else None
    return s or None


def _money(s: Optional[str]) -> tuple[Optional[float], bool]:
    """EDGAR money field -> (value, was_indefinite).

    The XML stores plain integers ("750000000"); the one legal non-number
    is the literal string "Indefinite" (unlimited offering size).
    """
    if s is None:
        return None, False
    if s.strip().lower() == "indefinite":
        return None, True
    try:
        return float(s.replace(",", "")), False
    except ValueError:
        return None, False   # unreadable -> None; caller flags it


def parse_form_d(xml_text: str, *, accession: str,
                 filing_date: Optional[str] = None) -> FormDFiling:
    """Parse one Form D primary_doc.xml. Deterministic; raises on broken XML.

    `accession` and `filing_date` come from the EDGAR submissions index (the
    driver knows them; the XML body does not repeat its own accession), so
    they are passed in rather than parsed out.

    Raising (rather than returning a half-record) on malformed XML is
    deliberate: a truth table must be complete-or-loudly-broken, never
    silently partial -- the caller counts every failure.
    """
    root = ET.fromstring(xml_text)      # ET.ParseError propagates on purpose

    flags: list[str] = []

    # --- who filed ---------------------------------------------------------
    cik = _text(root, "./primaryIssuer/cik") or ""
    name = _text(root, "./primaryIssuer/entityName") or ""
    if not cik:
        flags.append("missing_cik")
    if not name:
        flags.append("missing_entity_name")

    # --- new filing or amendment ------------------------------------------
    amend_node = "./offeringData/typeOfFiling/newOrAmendment"
    is_amend = (_text(root, amend_node + "/isAmendment") or "").lower() == "true"
    prev = _text(root, amend_node + "/previousAccessionNumber")
    if prev and not _ACCESSION_RX.fullmatch(prev.replace(" ", "")):
        flags.append("odd_previous_accession")

    # --- the round's date --------------------------------------------------
    # Either <dateOfFirstSale><value>ISO</value></dateOfFirstSale> or
    # <yetToOccur>true</yetToOccur> (no sale has happened yet).
    dfs = _text(root, "./offeringData/typeOfFiling/dateOfFirstSale/value")
    yet = (_text(root, "./offeringData/typeOfFiling/dateOfFirstSale/yetToOccur")
           or "").lower() == "true"
    if dfs is None and not yet:
        flags.append("missing_date_of_first_sale")

    # --- the money ---------------------------------------------------------
    sales = "./offeringData/offeringSalesAmounts"
    sold, sold_indef = _money(_text(root, sales + "/totalAmountSold"))
    offered, off_indef = _money(_text(root, sales + "/totalOfferingAmount"))
    if sold is None:
        # totalAmountSold is never legitimately "Indefinite"; record loudly.
        flags.append("unreadable_amount_sold" if not sold_indef
                     else "indefinite_amount_sold")

    return FormDFiling(
        accession=accession,
        cik=str(cik).zfill(10) if cik else "",
        entity_name=name,
        is_amendment=is_amend,
        previous_accession=prev,
        filing_date=filing_date,
        date_of_first_sale=dfs,
        amount_sold=sold,
        offering_amount=offered,
        indefinite=off_indef,
        parse_flags=tuple(flags),
    )


# ===========================================================================
# 15.2  The deterministic extract_fn (the oracle's "reader")
# ===========================================================================

def _accession_from_source(src: Source) -> str:
    """Recover the accession number the driver stamped on the Source.

    The driver sets title = f"{form} {accession}" and puts the dashless
    accession in the URL path; we take the title first (exact), the URL as
    a fallback, and "" if neither matches -- the parser then flags it.
    """
    m = _ACCESSION_RX.search(str(src.title or ""))
    if m:
        return m.group(0)
    # NOTE: Source.url is a pydantic HttpUrl, not a str -- coerce before regex
    m = re.search(r"/(\d{18})/", str(src.url or ""))
    if m:                                # re-hyphenate 18 digits: 10-2-6
        d = m.group(1)
        return f"{d[:10]}-{d[10:12]}-{d[12:]}"
    return ""


def _filing_date_from_source(src: Source) -> Optional[str]:
    """The driver parsed filingDate into Source.publish_time; back to ISO."""
    return (src.publish_time.strftime("%Y-%m-%d")
            if src.publish_time else None)


def parse_source(src: Source) -> FormDFiling:
    """Source (as built by EDGARDriver.fetch_for_key) -> parsed filing."""
    return parse_form_d(src.main_text,
                        accession=_accession_from_source(src),
                        filing_date=_filing_date_from_source(src))


def formd_mentions(src: Source, query: str) -> list[Mention]:
    """The DETERMINISTIC extract_fn for run_schema_addressable(as_oracle=True).

    Same (Source, query) -> list[Mention] contract as the LLM extractor, so
    the oracle sweep stores fully auditable provenance in {run}_truth.sqlite
    -- but every value here is an XML tag lookup: self_conf=1.0,
    extractor_id="oracle", accepted=True (no conformal gate needed; there is
    no distribution over readings to calibrate, delta_E is structurally 0).

    Emits per filing: amount (USD), date, and registry_key (the accession),
    all under record_kind="funding_round". The passage stored for each is
    the literal XML snippet that carries the value -- verbatim provenance,
    same standard as the open-web path.
    """
    filing = parse_source(src)
    now = datetime.utcnow()             # repo convention: UTC-naive throughout

    def mk(attribute: str, value: str, value_num: Optional[float],
           passage: str) -> Mention:
        return Mention(
            mention_id=Mention.make_id(src.source_id, filing.entity_name,
                                       "funding_round", attribute, value,
                                       "oracle", passage),
            source_id=src.source_id,
            entity_surface=filing.entity_name,
            record_kind="funding_round",
            attribute=attribute,
            value=value,
            passage=passage[:500],
            extracted_at=now,
            t_asof=src.publish_time,    # the fact holds as of the filing date
            value_num=value_num,
            currency="USD" if value_num is not None else None,
            extractor_id="oracle",
            self_conf=1.0,              # a tag lookup does not guess
            accepted=True,              # no gate: deterministic read
        )

    out: list[Mention] = []
    if filing.amount_sold is not None:
        v = (str(int(filing.amount_sold))
             if float(filing.amount_sold).is_integer()
             else str(filing.amount_sold))   # canonical numeric string form
        out.append(mk("amount", v, filing.amount_sold,
                      f"<totalAmountSold>{v}</totalAmountSold>"))
    if filing.date_of_first_sale:
        out.append(mk("date", filing.date_of_first_sale, None,
                      f"<dateOfFirstSale><value>{filing.date_of_first_sale}"
                      f"</value></dateOfFirstSale>"))
    if filing.accession:
        out.append(mk("registry_key", filing.accession, None,
                      f"accession {filing.accession}"))
    return out


# ===========================================================================
# 15.3  Chain collapse: filings -> one TruthRecord per round
# ===========================================================================

def collapse_chains(filings: list[FormDFiling]) -> list[dict]:
    """Group filings into amendment chains; keep each chain's NEWEST numbers.

    A chain is one funding round told over time: D, then zero or more D/A's.
    Linking uses previousAccessionNumber (exact, per-filing); a D/A whose
    predecessor was not fetched still roots its own (partial) chain and is
    flagged, never dropped -- its amounts are the freshest we have.

    Returns one plain dict per round (json-ready), sorted by date:
      {key, amount, date, accession, root_accession, n_filings,
       entity_name, flags}
    -- the caller turns these into frozen TruthRecords; keeping dicts here
    makes the metadata (which TruthRecord deliberately omits) storable.
    """
    by_acc = {f.accession: f for f in filings if f.accession}

    # UNION-FIND-lite: walk each filing back to its chain root. Chains are
    # a few links long, so the simple walk is plenty.
    def root_of(f: FormDFiling) -> str:
        seen = set()
        cur = f
        while (cur.previous_accession and cur.previous_accession in by_acc
               and cur.accession not in seen):
            seen.add(cur.accession)
            cur = by_acc[cur.previous_accession]
        # a D/A pointing OUTSIDE the fetched set roots a partial chain at
        # itself -- prefer its stated predecessor id as the stable root name
        # so re-runs that DO fetch the predecessor land on the same root.
        if cur.previous_accession and cur.previous_accession not in by_acc:
            return cur.previous_accession
        return cur.accession

    chains: dict[str, list[FormDFiling]] = {}
    for f in filings:
        chains.setdefault(root_of(f), []).append(f)

    rounds: list[dict] = []
    for root, members in chains.items():
        # newest filing wins -- the registry's own supersession rule.
        # Sort by (filing_date, accession): the accession tie-break makes
        # same-day amendment ordering deterministic.
        members = sorted(members, key=lambda f: (f.filing_date or "",
                                                 f.accession))
        last = members[-1]

        flags = sorted({fl for f in members for fl in f.parse_flags})
        if any(f.previous_accession not in by_acc
               for f in members if f.previous_accession):
            flags.append("partial_chain")   # predecessor outside the sweep

        # the round's date: first non-null dateOfFirstSale along the chain
        # (the ORIGINAL filing's, normally) -- it is the announcement-adjacent
        # date the open web reports; amendments rarely change it.
        date = next((f.date_of_first_sale for f in members
                     if f.date_of_first_sale), None)

        amount = last.amount_sold
        if amount is None:
            amount = 0.0                    # never guess; score the shortfall
            flags.append("amount_scored_zero")

        # the alignment key (matches risk_control.default_truth_key exactly):
        # kind|ISO-date when a date exists, else the accession -- still
        # matchable through the explicit registry_key attribute path.
        key = f"funding_round|{date}" if date else last.accession

        rounds.append({
            "key": key,
            "amount": float(amount),
            "date": date,
            "accession": last.accession,      # newest = authoritative numbers
            "root_accession": root,
            "n_filings": len(members),
            "entity_name": last.entity_name,
            "flags": flags,
        })

    rounds.sort(key=lambda r: (r["date"] or "9999", r["accession"]))

    # Duplicate-key check: two DISTINCT rounds sharing one first-sale date
    # would collide under kind|date matching. Rare but possible; grading
    # must know, so both records get flagged (they stay separate records).
    from collections import Counter
    dupes = {k for k, n in Counter(r["key"] for r in rounds).items() if n > 1}
    for r in rounds:
        if r["key"] in dupes:
            r["flags"] = sorted(set(r["flags"]) | {"duplicate_truth_key"})
    return rounds


def build_truth_entity(entity_id: str,
                       filings: list[FormDFiling]) -> tuple[TruthEntity, dict]:
    """Filings of ONE filer -> (TruthEntity, metadata dict).

    The TruthEntity is the frozen answer key risk_control grades against
    (true SUM and COUNT fall out of its properties); the metadata dict is
    everything else worth keeping (accessions, flags, per-round detail) and
    is what gets written to disk alongside it.
    """
    rounds = collapse_chains(filings)
    truth = TruthEntity(
        entity_id=entity_id,
        records=tuple(TruthRecord(key=r["key"], amount=r["amount"],
                                  date=r["date"]) for r in rounds),
    )
    meta = {
        "entity_id": entity_id,
        "entity_name": next((f.entity_name for f in filings
                             if f.entity_name), ""),
        "n_filings": len(filings),
        "n_rounds": truth.true_count,
        "true_sum": truth.true_sum,
        "true_count": truth.true_count,
        "rounds": rounds,
    }
    return truth, meta


# ===========================================================================
# 15.4  Disk format (data/ground_truth/<cohort>/)
# ===========================================================================
# One JSON per entity + one manifest per cohort. JSON (not sqlite) because
# the truth table is small, human-auditable by design, and diff-able in git.

def save_truth_entity(cohort_dir: Path, meta: dict) -> Path:
    """Write one entity's answer key; returns the path written."""
    cohort_dir.mkdir(parents=True, exist_ok=True)
    path = cohort_dir / f"truth_{meta['entity_id']}.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def load_truth_entity(path: Path) -> tuple[TruthEntity, dict]:
    """Read one entity JSON back into the frozen TruthEntity + its metadata."""
    meta = json.loads(Path(path).read_text())
    truth = TruthEntity(
        entity_id=meta["entity_id"],
        records=tuple(TruthRecord(key=r["key"], amount=float(r["amount"]),
                                  date=r.get("date"))
                      for r in meta["rounds"]),
    )
    return truth, meta


def load_truth_cohort(cohort_dir: Path) -> dict[str, TruthEntity]:
    """All entities of a cohort, keyed by entity_id (grading + LTT input)."""
    out: dict[str, TruthEntity] = {}
    for p in sorted(Path(cohort_dir).glob("truth_*.json")):
        truth, _ = load_truth_entity(p)
        out[truth.entity_id] = truth
    return out
