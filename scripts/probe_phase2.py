"""Phase 2 feasibility probe (dev-time, CI-run): real record counts and
response schemas from the two candidate oracles, so the Phase 2 prereg
quotes measured n's, not recollection.

  * ClinicalTrials.gov API v2 — trial counts for candidate queries + the
    field names of one study record;
  * OpenAlex — citation counts for candidate landmark papers + the field
    names of one work record.

Writes data/ci/phase2_probe.md (human) and .json (machine). Read-only,
no keys, ~a dozen paced requests. Loud on any unexpected shape.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "ci"
UA = {"User-Agent": "webagg-sports phase2 feasibility probe (research)"}

CT_BASE = "https://clinicaltrials.gov/api/v2/studies"
CT_QUERIES = [
    ("intervention: pembrolizumab", {"query.intr": "pembrolizumab"}),
    ("intervention: metformin", {"query.intr": "metformin"}),
    ("intervention: semaglutide", {"query.intr": "semaglutide"}),
    ("condition: breast cancer", {"query.cond": "breast cancer"}),
    ("cond breast cancer, phase 3 only",
     {"query.cond": "breast cancer", "filter.advanced": "AREA[Phase]PHASE3"}),
]

OA_BASE = "https://api.openalex.org/works"
OA_PAPERS = [
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "Attention Is All You Need",
    "Deep Residual Learning for Image Recognition",
]


def get(url, params):
    time.sleep(1.0)
    r = requests.get(url, params=params, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> int:
    report = {"clinicaltrials": [], "openalex": []}
    lines = ["# Phase 2 feasibility probe", ""]

    lines.append("## ClinicalTrials.gov v2 (oracle candidate: trials)")
    for label, params in CT_QUERIES:
        data = get(CT_BASE, {**params, "countTotal": "true", "pageSize": 1})
        total = data.get("totalCount")
        if total is None:
            raise SystemExit(f"no totalCount for {label}: keys={list(data)}")
        report["clinicaltrials"].append({"query": label, "total": total})
        lines.append(f"- {label}: **{total}** registered studies")
    study = data["studies"][0]
    ps = study.get("protocolSection", {})
    schema = {sec: sorted(v.keys()) for sec, v in ps.items()
              if isinstance(v, dict)}
    report["ct_schema_sections"] = schema
    lines += ["", "One study's protocolSection layout:",
              "```json", json.dumps(schema, indent=1)[:1500], "```", ""]

    lines.append("## OpenAlex (oracle candidate: citations)")
    for title in OA_PAPERS:
        data = get(OA_BASE, {"search": title, "per-page": 1})
        results = data.get("results") or []
        if not results:
            raise SystemExit(f"no OpenAlex result for {title!r}")
        w = results[0]
        row = {"paper": title, "matched": w.get("display_name"),
               "id": w.get("id"), "cited_by_count": w.get("cited_by_count"),
               "year": w.get("publication_year")}
        report["openalex"].append(row)
        lines.append(f"- {title!r} -> matched {row['matched']!r} "
                     f"({row['year']}): **{row['cited_by_count']}** citations"
                     f" — {row['id']}")
    report["oa_work_fields"] = sorted(results[0].keys())
    lines += ["", f"Work record fields: {report['oa_work_fields']}", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase2_probe.json").write_text(json.dumps(report, indent=1))
    (OUT / "phase2_probe.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
