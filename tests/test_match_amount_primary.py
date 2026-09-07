"""Amount-primary grading key (A3 pre-registration Sec. 3).

Press dates are announcement dates; truth keys carry filing/firstSale
dates. The pre-registered key therefore aligns by amount first:
registry key -> nearest amount within tolerance (date breaks exact ties)
-> exact date key as last resort. Legacy behavior (amount_primary=False)
must stay byte-for-byte identical.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg.type_defs import CorroboratedValue
from webagg.risk_control import TruthEntity, TruthRecord, match_to_truth

TOL = 0.05


class FakeRecord:
    """Same contract slice as tests/test_risk_control_ch13.py pins."""
    def __init__(self, kind="funding_round", **attrs):
        self.record_kind = kind
        self.attributes = {
            k: CorroboratedValue(value=str(v), belief=0.9, nu=1,
                                 component_sizes=[1],
                                 value_num=(float(v) if isinstance(v, (int, float))
                                            else None))
            for k, v in attrs.items()}


def _truth(*rounds):
    """rounds: (amount, key, date) triples."""
    recs = tuple(TruthRecord(key=k, amount=a, date=d) for a, k, d in rounds)
    return TruthEntity(entity_id="acme", records=recs)


def test_amount_primary_requires_tolerance():
    with pytest.raises(ValueError):
        match_to_truth([], _truth(), amount_primary=True)  # loud, not silent


def test_amount_beats_a_misleading_exact_date():
    # press dated the $40M announcement on the same day another round FILED
    truth = _truth((40e6, "funding_round|2021-03-28", "2021-03-28"),
                   (10e6, "funding_round|2020-06-17", "2020-06-17"))
    rec = FakeRecord(amount=40e6, date="2020-06-17")
    # legacy: exact date key grabs the WRONG ($10M) round
    (_, t_legacy), = match_to_truth([rec], truth)
    assert t_legacy.amount == 10e6
    # amount-primary: the $40M finds the $40M
    (_, t_new), = match_to_truth([rec], truth, amount_tol=TOL,
                                 amount_primary=True)
    assert t_new.amount == 40e6


def test_exact_amount_tie_goes_to_nearest_date():
    # two truth rounds with IDENTICAL amounts, years apart (the collision
    # case Sec. 3 counted: 5/672 pairs; date separates them)
    truth = _truth((20e6, "funding_round|2019-05-01", "2019-05-01"),
                   (20e6, "funding_round|2023-01-10", "2023-01-10"))
    rec = FakeRecord(amount=20e6, date="2023-01-04")
    (_, t), = match_to_truth([rec], truth, amount_tol=TOL,
                             amount_primary=True)
    assert t.key == "funding_round|2023-01-10"


def test_registry_key_still_wins_over_amount():
    truth = _truth((40e6, "0001-23-000045", "2021-03-28"),
                   (41e6, "funding_round|2022-08-09", "2022-08-09"))
    # amount 41e6 is nearest the SECOND round, but the accession is explicit
    rec = FakeRecord(amount=41e6, registry_key="0001-23-000045")
    (_, t), = match_to_truth([rec], truth, amount_tol=TOL,
                             amount_primary=True)
    assert t.key == "0001-23-000045"


def test_no_amount_falls_back_to_exact_date_key():
    truth = _truth((40e6, "funding_round|2021-03-28", "2021-03-28"))
    rec = FakeRecord(date="2021-03-28")          # no readable amount
    (_, t), = match_to_truth([rec], truth, amount_tol=TOL,
                             amount_primary=True)
    assert t is not None and t.amount == 40e6


def test_one_to_one_still_punishes_oversplits():
    truth = _truth((40e6, "funding_round|2021-03-28", "2021-03-28"))
    dup = [FakeRecord(amount=40e6, date="2021-03-15"),
           FakeRecord(amount=40e6, date="2021-03-16")]
    aligned = match_to_truth(dup, truth, amount_tol=TOL, amount_primary=True)
    matched = [t for _, t in aligned if t is not None]
    assert len(matched) == 1                     # the duplicate stays spurious


def test_beyond_tolerance_stays_spurious():
    truth = _truth((40e6, "funding_round|2021-03-28", "2021-03-28"))
    rec = FakeRecord(amount=50e6, date="2020-01-01")   # 25% off, wrong date
    (_, t), = match_to_truth([rec], truth, amount_tol=TOL,
                             amount_primary=True)
    assert t is None
