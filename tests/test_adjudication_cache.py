"""The per-certification adjudication cache: one verdict per pair, order-
independent keys, and the base adjudicator is only consulted on misses."""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webagg.certify import memoized_adjudicator


@dataclass
class _M:
    mention_id: str
    entity_surface: str = "X"


def test_pair_judged_once_and_order_independent():
    calls = []

    def base(a, b, lookup):
        calls.append((a.mention_id, b.mention_id))
        return 0.9

    cache: dict = {}
    adj = memoized_adjudicator(cache, base=base)
    a, b = _M("m1"), _M("m2")
    assert adj(a, b, {}) == 0.9
    assert adj(b, a, {}) == 0.9          # reversed order -> same key
    assert adj(a, b, {}) == 0.9
    assert len(calls) == 1               # base consulted exactly once
    assert cache == {("m1", "m2"): 0.9}


def test_distinct_pairs_get_distinct_verdicts():
    def base(a, b, lookup):
        return 0.8 if "3" in b.mention_id else 0.2

    cache: dict = {}
    adj = memoized_adjudicator(cache, base=base)
    assert adj(_M("m1"), _M("m2"), {}) == 0.2
    assert adj(_M("m1"), _M("m3"), {}) == 0.8
    assert len(cache) == 2
