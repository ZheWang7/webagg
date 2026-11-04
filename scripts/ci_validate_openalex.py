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


if __name__ == "__main__":
    raise SystemExit(main())
