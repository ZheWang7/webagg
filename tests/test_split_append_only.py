"""Append-only split regression (instrument fix, pre-freeze).

The calibration/validation split is history, not a computation: once an
entity is dealt to a side it never moves, however the cohort later grows
(a top-up) or shrinks (a late audit flip). Re-dealing would move entities
across the calibration/validation line and void the held-out check behind
eps_F (design doc Sec. 13.3, practical rule 2).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg.risk_control import assign_split


def _cohort(n=52):
    return [f"cik{i:04d}" for i in range(n)]


def test_fresh_deal_is_deterministic_and_hits_target():
    ids = _cohort()
    cal1, val1, d1 = assign_split(ids, seed=0, cal_frac=2 / 3)
    cal2, val2, d2 = assign_split(ids, seed=0, cal_frac=2 / 3)
    assert (cal1, val1) == (cal2, val2)        # seeded -> reproducible
    assert d1 == [] and d2 == []
    assert len(cal1) == int(52 * 2 / 3) == 34  # the 2/3-heavy split: 34/18
    assert set(cal1) | set(val1) == set(ids)   # everyone dealt exactly once
    assert not set(cal1) & set(val1)


def test_additions_never_move_existing_assignments():
    ids = _cohort()
    cal1, val1, _ = assign_split(ids, seed=0, cal_frac=2 / 3)
    grown = ids + ["cik9001", "cik9002", "cik9003"]
    cal2, val2, dropped = assign_split(
        grown, seed=0, cal_frac=2 / 3,
        prior={"calibration": cal1, "validation": val1})
    assert set(cal1) <= set(cal2)              # nobody moved sides
    assert set(val1) <= set(val2)
    assert dropped == []
    assert set(cal2) | set(val2) == set(grown)
    assert not set(cal2) & set(val2)


def test_removals_drop_loudly_and_move_nobody_else():
    ids = _cohort()
    cal1, val1, _ = assign_split(ids, seed=0, cal_frac=2 / 3)
    victim = cal1[0]                           # e.g. a late audit flip
    shrunk = [i for i in ids if i != victim]
    cal2, val2, dropped = assign_split(
        shrunk, seed=0, cal_frac=2 / 3,
        prior={"calibration": cal1, "validation": val1})
    assert dropped == [victim]                 # departure is reported
    assert cal2 == sorted(set(cal1) - {victim})
    assert val2 == val1                        # the other side untouched
