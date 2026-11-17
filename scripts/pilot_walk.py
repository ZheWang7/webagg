"""Exp 2-C PILOT 2: a short LIVE index walk on a non-cohort paper
(PREREGISTRATION_EXP2C rev 3 §6.4). Cheap by design: it is capped at
--max-steps capture occasions, not run to certification.

What it establishes before the cohort is frozen:
  1. the plug-in works live: pages fetch `ok`, cursors advance, and no
     occasion ever credits more than the page size (Assumption 1b);
  2. the realized per-page novelty and cross-index re-coverage so far;
  3. the conformal gate, WITHOUT hand labeling: the first index pages are
     also read by the LLM reader, and each LLM row is labeled correct iff
     its DOI key is among the rows the API itself returned for that page.
     Result: data/conformal/calibration.csv (needs >= 20 correct rows).

Usage:
    $env:RUN_LIVE="1"
    python scripts/pilot_walk.py --title "Highly accurate protein structure prediction with AlphaFold" `
        --doi 10.1038/s41586-021-03819-2 --pmid 34265844 --cutoff 2023-12-31
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webaggs import config                      # noqa: E402
from webaggs.conformal import ConformalGate     # noqa: E402
from webaggs.indexes import Target, make_indexes  # noqa: E402
from webaggs.storage import fresh_ledger        # noqa: E402
from webaggs.types import Source                # noqa: E402

OUT = config.DATA_DIR / "pilot_exp2c"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True,
                    help="pilot paper TITLE — must NOT be in the cohort")
    ap.add_argument("--doi", required=True)
    ap.add_argument("--pmid", default=None)
    ap.add_argument("--cutoff", required=True, help="YYYY-12-31")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--indexes", default="openalex,europepmc")
    ap.add_argument("--page-size", type=int, default=config.INDEX_PAGE_SIZE)
    ap.add_argument("--calibration-pages", type=int, default=2,
                    help="index pages per index also read by the LLM reader")
    args = ap.parse_args()

    if not config.live_enabled():
        raise SystemExit("RUN_LIVE != 1 — refusing live spend.")
    for env in (config.OPENAI_API_KEY_ENV, config.SERPER_API_KEY_ENV):
        if not os.environ.get(env):
            raise SystemExit(f"{env} not set — refusing to run.")

    from webaggs.canon import normalize_title   # noqa: E402
    from webaggs.extract import extract_certified  # noqa: E402
    from webaggs.pipeline import run_query      # noqa: E402
    from webaggs.search import SerperBackend    # noqa: E402

    permissive = ConformalGate(threshold=1.0, n_cal=0)   # label first, gate later
    target = Target(title=args.title, doi=args.doi, pmid=args.pmid)
    names = [n.strip() for n in args.indexes.split(",") if n.strip()]
    rid = "pilot2_" + normalize_title(args.title).replace(" ", "_")[:40]
    ledger = fresh_ledger(OUT / "runs", rid)
    print(f"[live] pilot walk on {args.title!r}: {args.max_steps} occasions, "
          f"page size {args.page_size}, indexes {names} ...")
    state, art = run_query(args.title, rid, ledger, search=SerperBackend(),
                           gate=permissive, budget_usd=args.budget,
                           max_steps=args.max_steps, target=target,
                           indexes=make_indexes(names), cutoff=args.cutoff,
                           page_size=args.page_size, y_cap=args.page_size)

    # --- 1. plumbing: outcomes per index domain, novelty bound -----------
    print(f"\nstop: {art['stop_reason']} at step {art['stop_step']}")
    for dom, st, n in ledger.conn.execute(
            "SELECT s.domain, s.status, COUNT(*) FROM sources s "
            "WHERE s.source_class='citation_index' GROUP BY 1, 2"):
        print(f"  index pages  {dom:28s} {st:8s} x{n}")
    for (why,) in ledger.conn.execute(
            "SELECT extra FROM measurements WHERE metric='index_unavailable'"):
        print(f"  INDEX UNAVAILABLE: {why}")
    per_step = Counter(s for s, _ in ledger.coverage(rid))
    worst = max(per_step.values()) if per_step else 0
    print(f"max new records in one occasion: {worst} "
          f"(bound = page size {args.page_size}: "
          f"{'OK' if worst <= args.page_size else 'VIOLATED'})")

    # --- 2. what the walk has seen so far --------------------------------
    pool = {next(iter(state.strata))}
    n, f1 = state.N(pool), state.f(1, pool)
    print(f"records so far: N={n}, singletons f1={f1} "
          f"(share {f1 / max(n, 1):.2f} — only meaningful after BOTH walks "
          "finish; the probe measures the real overlap)")

    # --- 3. conformal calibration, auto-labeled against the API rows -----
    rows = []
    srcs = ledger.conn.execute(
        "SELECT source_id, url, domain, main_text FROM sources "
        "WHERE source_class='citation_index' AND status='ok' "
        "ORDER BY fetch_time").fetchall()
    taken = Counter()
    for sid, url, dom, text in srcs:
        if taken[dom] >= args.calibration_pages:
            continue
        taken[dom] += 1
        # the page's own typed rows = the labels for the LLM reader
        api_keys = {k for (k,) in ledger.conn.execute(
            "SELECT season FROM mentions WHERE source_id=?", (sid,))}
        src = Source(source_id=f"cal-{sid}", url=url, domain=dom,
                     fetch_time="", formulation_id="cal", status="ok",
                     main_text=text, source_class="citation_index")
        ms, _claims, _rej = extract_certified(
            src, args.title, permissive, ledger=ledger, run_id=rid,
            require_doi=False)
        for m in ms:
            rows.append((m.confidence, int(m.season in api_keys), m.season,
                         url))
    ledger.commit()
    ledger.close()
    n_ok = sum(c for _conf, c, *_x in rows)
    cal_dir = config.DATA_DIR / "conformal"
    cal_dir.mkdir(parents=True, exist_ok=True)
    with open(cal_dir / "calibration.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "correct", "cite_key", "source_url"])
        w.writerows(rows)
    print(f"\nconformal calibration: {len(rows)} LLM-read rows, {n_ok} "
          f"correct -> {cal_dir / 'calibration.csv'}")
    if n_ok < 20:
        print("  fewer than 20 correct rows: raise --calibration-pages "
              "and re-run (the gate refuses to fit below 20).")
    else:
        gate = ConformalGate.load()
        print(f"  gate fits: threshold {gate.threshold:.3f}, n_cal {gate.n_cal}")


if __name__ == "__main__":
    main()
