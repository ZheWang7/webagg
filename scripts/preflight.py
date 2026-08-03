"""Pre-flight probe: is a query's web coverage FETCHABLE, before you spend
a long run on it?

The ch.-14 live runs kept starving for the same reason: funding-news
domains stonewall plain HTTP clients, so pages die at the fetch stage and
discovery sees one record in 25 steps. This probe answers the question
cheaply and in advance: ONE Serper search (~$0.02) + up to k fetches
through the pipeline's OWN fetch_url (same UA, same politeness, same
non-content filter), NO LLM calls. Per URL it prints the fate the pipeline
would see; at the end, a survival rate and the source-class mix.

Reading the output:
  * survival < ~30%  -> pick different entities; a long run will starve.
  * anchored > 0     -> registry-class pages reached (identity_anchored:
                        these escape the qbar cap, so beliefs can be high).
  * classes          -> a mix (news + vendor + registry) is what the
                        fragmentation and corroboration machinery feeds on.

Usage (repo root; .env needs SERPER_API_KEY only):

    python scripts/preflight.py "funding rounds of Mistral AI"
    python scripts/preflight.py "Cohere funding history" -k 10
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg.fetch import clear_fetch_cache, fetch_url    # noqa: E402
from webagg.search import SerperBackend                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", help="the query you are considering running")
    ap.add_argument("-k", type=int, default=10,
                    help="how many search results to probe (default 10)")
    args = ap.parse_args()

    clear_fetch_cache()
    results = SerperBackend().search(args.query, k=args.k,
                                     formulation_id="preflight")
    if not results:
        sys.exit("search returned nothing -- check SERPER_API_KEY / query")

    survived, classes, anchored = 0, Counter(), 0
    print(f"\nprobing {len(results)} results for: {args.query!r}\n")
    for r in results:
        src = fetch_url(r["url"], formulation_id="preflight")
        if src is None:
            # same verdict the pipeline would reach: unreachable, non-HTML,
            # non-200, or a paywall/login stub under 200 chars of text
            print(f"  DEAD      {r['url'][:90]}")
            continue
        survived += 1
        classes[src.source_class or "other"] += 1
        anchored += bool(src.identity_anchored)
        print(f"  OK  {src.source_class or 'other':<11}"
              f"{'ANCHORED ' if src.identity_anchored else '         '}"
              f"{len(src.main_text):>6} chars  {r['url'][:70]}")

    rate = survived / len(results)
    print(f"\nsurvival: {survived}/{len(results)} ({rate:.0%})   "
          f"anchored: {anchored}   classes: {dict(classes)}")
    if rate < 0.3:
        print("verdict: STARVED -- a long run on this query will mostly "
              "fetch dead pages; try entities with Wikipedia/newsroom/"
              "registry coverage instead.")
    else:
        print("verdict: fetchable -- worth a long run.")


if __name__ == "__main__":
    main()
