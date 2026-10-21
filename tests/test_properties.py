"""Property / metamorphic tests: invariants that must hold over RANDOM inputs,
not just the hand-picked cases. Fixed seeds keep every run reproducible."""
import random
import string

import pytest

from webaggs import canon
from webaggs.claims import ClaimsEngine
from webaggs.corroborate import corroborate_record
from webaggs.er import resolve_entity
from webaggs.frontier import Formulation, FrontierState, StratumState, \
    all_strata_pass
from webaggs.replay import replay_stop, stop_predicate
from webaggs.types import Claim, Mention, Source


# ---------------------------------------------------------------------------
# THE property Exp 2 stands on: the live stop test and the offline replay
# predicate must be the same function of the logged quantities. If these two
# ever drift, every looser-eps replay arm is invalid.
# ---------------------------------------------------------------------------
def _random_state(rng) -> FrontierState:
    st = FrontierState()
    st.strata["p"] = StratumState(name="p")
    for i in range(rng.randint(0, 40)):
        st.covered[f"p|s{i}"] = {f"fm{k}" for k in range(rng.randint(1, 4))}
        st.record_stratum[f"p|s{i}"] = "p"
    for i in range(rng.randint(0, 4)):
        st.formulations[f"q{i}"] = Formulation(
            f"q{i}", f"query {i}", p_success=rng.uniform(0, 1),
            yield_if_success=rng.uniform(0, 4),
            issued=rng.random() < 0.5, stratum=rng.choice(["p", None]))
    if rng.random() < 0.3:
        st.strata["p"].claimed_count = rng.randint(0, 45)
    if rng.random() < 0.2:
        st.strata["p"].certified = "checksum"
    st.Y_CAP = rng.choice([1, 4, 12])
    return st


def test_live_stop_test_equals_replay_predicate_over_random_states():
    rng = random.Random(11)
    delta_m, eta, max_steps = 0.05, 0.20, 25
    for _ in range(400):
        st = _random_state(rng)
        pool = {"p"}
        u, psi = st.U_hat(pool), st.psi(pool, delta_m, 1.0, max_steps)
        hot = st.hot_frontier("p", eta)
        s = st.strata["p"]
        for eps in (0.01, 0.05, 0.1, 0.3, 0.9, 2.0):
            live = all_strata_pass(st, eps, delta_m, eta, max_steps)
            rep = stop_predicate(u, psi, hot, st.N(pool), s.claimed_count,
                                 s.certified, eps)
            assert live == rep, (eps, u, psi, hot, s)


def test_f_counts_partition_the_records():
    rng = random.Random(13)
    for _ in range(100):
        st = _random_state(rng)
        n = st.N({"p"})
        assert sum(st.f(k, {"p"}) for k in range(1, 6)) == n


def test_psi_monotonicities():
    st = FrontierState()
    st.strata["p"] = StratumState(name="p")
    for i in range(50):
        st.covered[f"p|s{i}"] = {"a"}
        st.record_stratum[f"p|s{i}"] = "p"
    small = FrontierState()
    small.strata["p"] = StratumState(name="p")
    for i in range(5):
        small.covered[f"p|s{i}"] = {"a"}
        small.record_stratum[f"p|s{i}"] = "p"
    # radius shrinks with more records, grows as the deviation budget w_g shrinks
    assert st.psi({"p"}, 0.05, 1.0, 50) < small.psi({"p"}, 0.05, 1.0, 50)
    assert st.psi({"p"}, 0.05, 0.1, 50) > st.psi({"p"}, 0.05, 1.0, 50)


def test_replay_stop_monotone_in_eps_over_random_traces():
    rng = random.Random(17)
    for _ in range(200):
        rows = []
        for step in range(1, rng.randint(2, 30)):
            rows.append((step, "p", "U_hat", rng.uniform(0, 1.2),
                         {"psi": rng.uniform(0, 0.6),
                          "hot": rng.random() < 0.4,
                          "N": rng.randint(0, 40),
                          "claimed_count": rng.choice([None, rng.randint(0, 45)]),
                          "certified": rng.choice(["", "", "", "checksum"])}))
        stops = [replay_stop(rows, e).stop_step for e in (0.05, 0.1, 0.2, 0.3)]
        prev = None
        for s in stops:                      # looser eps: stop never later
            if prev is not None:
                assert s is not None and s <= prev
            prev = s if s is not None else prev


# ---------------------------------------------------------------------------
def _mention(value, src_id):
    return Mention(mention_id=f"m{src_id}{value}", source_id=src_id,
                   entity_surface="P", season="2019-20", attribute="points",
                   value=value, confidence=0.9)


def _source(src_id, domain, cls="news"):
    return Source(source_id=src_id, url=f"https://{domain}/x", domain=domain,
                  fetch_time="t", formulation_id="f", status="ok",
                  source_class=cls)


def test_corroboration_echo_invariance_property():
    """Duplicating same-domain evidence never changes the adopted value or
    belief; new-domain support for the winner never lowers belief (§4.3)."""
    sources = {f"s{i}": _source(f"s{i}", f"d{i}.test") for i in range(6)}
    base = [_mention(100, "s0"), _mention(100, "s1"), _mention(200, "s2")]
    r0 = corroborate_record("p|2019-20", "2019-20", base, sources)
    echoed = base + [_mention(200, "s2")] * 25        # echo: same domain
    r1 = corroborate_record("p|2019-20", "2019-20", echoed, sources)
    assert (r0.value, r0.belief) == (r1.value, r1.belief)
    boosted = base + [_mention(100, "s3")]            # fresh domain for winner
    r2 = corroborate_record("p|2019-20", "2019-20", boosted, sources)
    assert r2.value == 100 and r2.belief >= r0.belief


def test_claims_corroboration_is_ingestion_order_invariant():
    rng = random.Random(19)
    claims = ([Claim(f"a{i}", "s", f"d{i}.t", "p", "COUNT", 3) for i in range(3)]
              + [Claim(f"b{i}", "s", f"e{i}.t", "p", "COUNT", 5) for i in range(2)])
    results = set()
    for _ in range(20):
        rng.shuffle(claims)
        ce = ClaimsEngine()
        for c in claims:
            ce.ingest(c)
        results.add(ce.corroborated("p", "COUNT"))
    assert results == {3}                             # strict winner, any order


def test_er_is_invariant_to_case_diacritics_spacing():
    target = "Luka Dončić"
    for surf in ("luka doncic", "LUKA  DONČIĆ", " Luka Dončić ", "L. Doncic",
                 "Doncic"):
        assert resolve_entity(surf, target) == "luka doncic", surf
    assert resolve_entity("Mario Doncic", target) is None


@pytest.mark.parametrize("seed", [23, 29, 31])
def test_canon_fuzz_never_crashes_and_outputs_are_typed(seed):
    rng = random.Random(seed)
    alphabet = string.printable + "–—é北京ćžš"
    for _ in range(500):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        season = canon.canonical_season(s)
        if season is not None:                        # shape: YYYY-YY, end=start+1
            start, end = int(season[:4]), int(season[5:7])
            assert len(season) == 7 and end == (start + 1) % 100
        pts = canon.canonical_points(s)
        assert pts is None or (isinstance(pts, int) and 0 <= pts <= 60000)
        name = canon.normalize_name(s)
        assert name == name.strip() and name.lower() == name
