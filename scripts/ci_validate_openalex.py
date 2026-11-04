"""CI live validation of the OpenAlex oracle driver (no keys, read-only):
exercises the REAL driver — work_meta plus a full citers() cursor walk
with self-validation on a finished early slice (papers citing 'Attention
Is All You Need' published before 2017-09-01). Writes data/ci/
exp2c_oracle_check.md; any schema drift or partial walk raises loudly."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs.oracle_openalex import OpenAlexDriver  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "ci"
WID = "W2626778328"                       # Attention Is All You Need


def main() -> int:
    d = OpenAlexDriver()
    meta = d.work_meta(WID)
    assert "attention is all you need" in meta["title"].lower(), meta
    assert meta["year"] == 2017 and meta["cited_by_count"] > 50000, meta

    rows = d.citers(WID, "2017-09-01")    # finished slice: full walk +
    keys = {r["key"] for r in rows}       # meta.count self-validation inside
    assert 0 < len(rows) < 2000 and len(keys) == len(rows)
    with_doi = sum(1 for r in rows if r["key"].startswith("doi:"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp2c_oracle_check.md").write_text(
        "# Exp 2-C oracle driver — live validation\n\n"
        f"- work_meta: {meta['title']!r} ({meta['year']}), lifetime "
        f"cited_by_count {meta['cited_by_count']}\n"
        f"- citers walk (cutoff 2017-09-01): **{len(rows)}** citers, "
        "cursor walk == meta.count (self-validated), all keys distinct\n"
        f"- key mix: {with_doi} doi-keyed / {len(rows) - with_doi} "
        "title-keyed\n")
    print((OUT / "exp2c_oracle_check.md").read_text())
    return 0


def _diagnose():
    """CI logs are unreachable from the dev sandbox, so the run carries
    its own diagnostics: on ANY failure the traceback plus raw API
    probes are committed back in the report before the nonzero exit."""
    import json
    import traceback

    import requests
    try:
        return main()
    except BaseException:
        OUT.mkdir(parents=True, exist_ok=True)
        probe = {}
        for label, params in (
            ("work_meta", {"filter": f"ids.openalex:{WID}",
                           "per-page": 1}),
            ("citers_page1", {"filter": f"cites:{WID},"
                                        "to_publication_date:2017-09-01",
                              "per-page": 3, "cursor": "*"})):
            try:
                r = requests.get("https://api.openalex.org/works",
                                 params=params, timeout=30)
                probe[label] = f"HTTP {r.status_code}: {r.text[:900]}"
            except Exception as e:  # noqa: BLE001
                probe[label] = f"EXC {type(e).__name__}: {e}"
        report = ("# Exp 2-C oracle validation FAILED\n\n```\n"
                  + traceback.format_exc()
                  + "\n```\n\nRaw probes:\n```json\n"
                  + json.dumps(probe, indent=1) + "\n```\n")
        (OUT / "exp2c_oracle_check_ERROR.md").write_text(report)
        raise


if __name__ == "__main__":
    raise SystemExit(_diagnose())
