"""
Live sanity check for schema-addressable mode (Section 9).

Runs REAL fetches against SEC EDGAR and ClinicalTrials.gov.
The point is to confirm the network plumbing works end to end:
the registries are reachable, the JSON shapes match what the drivers expect,
pagination resolves, and a real filing / trial comes back as a usable Source.
"""
import sys, os, pathlib

# Make `import webagg` work no matter where this is launched from, and make the
# CWD the repo root (only matters if you later flip on the full LLM runner,
# which reads prompts/ via a relative path).
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from webagg.schema_addressable import EDGARDriver, ClinicalTrialsDriver


def banner(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


def edgar_sanity() -> None:
    banner("SEC EDGAR  (live: www.sec.gov + data.sec.gov)")
    # The default client sends a User-Agent that carries an email, which SEC
    # requires -- a generic agent gets a 403.
    ed = EDGARDriver()

    # (1) ENUMERATION: prove the key universe K is enumerable. This downloads
    #     the published company index (~1 MB) and finds Tesla's CIK in it.
    print("Looking up Tesla in the published company index (~1 MB download)...")
    matches = list(ed.enumerate_keys({"name_contains": "tesla"}))
    print(f"  CIKs whose company name contains 'tesla': {matches}")

    # (2) FETCH mu(k): pull Tesla's recent 10-Ks for real. Each 10-K primary
    #     document is several MB, so this step downloads ~10-20 MB -- give it
    #     ~20 seconds. Lower 'since' (e.g. "2010") to see more history.
    CIK = "0001318605"  # Tesla, Inc.
    print(f"\nFetching 10-K filings for CIK {CIK} since 2024...")
    keys = list(ed.enumerate_keys({"ciks": [CIK], "forms": ["10-K"], "since": "2024"}))
    for cik in keys:
        srcs = ed.fetch_for_key(cik)
        print(f"  CIK {cik}: {len(srcs)} 10-K document(s) fetched")
        for s in srcs:
            when = s.publish_time.strftime("%Y-%m-%d") if s.publish_time else "?"
            print(f"   - {s.title}   filed {when}   domain={s.domain}")
            print(f"     {s.url}")
        if srcs:
            print("\n   --- first 400 chars of the most recent 10-K's text ---")
            snippet = srcs[0].main_text[:400].replace("\n", " ")
            print("   " + snippet)


def ctgov_sanity() -> None:
    banner("ClinicalTrials.gov  (live: clinicaltrials.gov/api/v2)")
    ct = ClinicalTrialsDriver(page_size=10)  # small page; we only need a few

    # (1) ENUMERATION + pagination: stream Phase 3 pembrolizumab trials. We stop
    #     after 3 ids so we only pull the first page (not the whole result set).
    print("Enumerating Phase 3 pembrolizumab trials (stopping after 3)...")
    keys = []
    for nct in ct.enumerate_keys({"intervention": "pembrolizumab", "phase": 3}):
        keys.append(nct)
        if len(keys) >= 3:
            break
    print(f"  first NCT ids: {keys}")

    # (2) FETCH mu(k): pull one full trial record and show the flattened text
    #     that the extractor would read.
    if keys:
        print(f"\nFetching the full record for {keys[0]}...")
        srcs = ct.fetch_for_key(keys[0])
        if srcs:
            s = srcs[0]
            print(f"  domain={s.domain}   title={s.title!r}")
            print("   --- flattened study text (what extract_mentions sees) ---")
            for line in s.main_text.splitlines()[:8]:
                print("   " + line)


def main() -> None:
    edgar_sanity()
    # ctgov_sanity()
    banner("DONE")
    print("If you saw real filing titles + dates + text, and real NCT ids +")
    print("trial text above, the live schema-addressable fetch path works.")


if __name__ == "__main__":
    main()
