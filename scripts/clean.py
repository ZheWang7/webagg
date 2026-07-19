"""Clear per-run artifacts: run DBs/logs and the on-disk HTML cache.

Usage (from the project root):
    python scripts/clean.py                # delete data/runs/* and data/cache/*
    python scripts/clean.py --dry-run      # show what WOULD be deleted, touch nothing
    python scripts/clean.py --keep tier2_sanity_01 demo_run   # spare named runs
    python scripts/clean.py --pycache      # also sweep __pycache__ / .pytest_cache

Cleans data/runs/ (per-run sqlite DBs + json logs), data/cache/ (fetch
cache), and the legacy html_cache/ if present. Never touches
data/ground_truth/ or data/calibration/ -- those are inputs (oracle
fixtures, conformal calibration sets), not run outputs.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

# Import the SAME path constants the pipeline uses (webagg/config.py), so
# this script can never drift from where runs are actually written.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webagg.config import RUNS_DIR, DATA_DIR, HTML_CACHE_DIR, ROOT_DIR  # noqa: E402

CACHE_DIRS = [DATA_DIR / "cache", HTML_CACHE_DIR]   # whichever exist


def human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def collect(keep: set[str]) -> list[Path]:
    """Everything under data/runs/ and html_cache/, minus kept run ids.

    A run 'tier2_sanity_01' may own several files (tier2_sanity_01.sqlite,
    tier2_sanity_01.json, ...): --keep matches on the file STEM so all of a
    kept run's artifacts survive together.
    """
    targets = []
    for f in sorted(RUNS_DIR.glob("*")):
        if f.stem in keep:
            continue
        targets.append(f)
    for cache_dir in CACHE_DIRS:
        if cache_dir.exists():               # skip silently if absent
            targets += sorted(cache_dir.glob("*"))
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be deleted; delete nothing")
    ap.add_argument("--keep", nargs="*", default=[], metavar="RUN_ID",
                    help="run ids to spare (matched against file stems)")
    ap.add_argument("--pycache", action="store_true",
                    help="also remove __pycache__/ and .pytest_cache/ trees")
    args = ap.parse_args()

    targets = collect(set(args.keep))
    if args.pycache:
        targets += sorted(ROOT_DIR.rglob("__pycache__"))
        targets += sorted(ROOT_DIR.rglob(".pytest_cache"))

    if not targets:
        print("Nothing to clean.")
        return

    freed = 0
    for t in targets:
        size = (sum(p.stat().st_size for p in t.rglob("*") if p.is_file())
                if t.is_dir() else t.stat().st_size)
        freed += size
        print(f"{'[dry-run] ' if args.dry_run else ''}rm {t.relative_to(ROOT_DIR)}"
              f"  ({human(size)})")
        if not args.dry_run:
            shutil.rmtree(t) if t.is_dir() else t.unlink()

    verb = "Would free" if args.dry_run else "Freed"
    print(f"\n{verb} {human(freed)} across {len(targets)} item(s).")
    # config.py re-creates data/runs/ and html_cache/ on next import, so an
    # empty tree after cleaning is fine.


if __name__ == "__main__":
    main()
