"""Randomized stress campaign: generated worlds through the REAL end_to_end
(fakes only for search/fetch/llm — tests only). Two arms:

  A. benign worlds (truthful claims, partial coverage, paywalls, name
     variants, contamination)
  B. adversarial worlds (a WRONG, corroborated COUNT claim above the truth)

Invariants checked on every run:
  I1  no exception; a stop is always recorded with a known reason
  I2  reported total == sum of resolved records
  I3  value correctness: every adopted season value equals truth (sources
      agree in these worlds, so corroboration must recover the truth)
  I4  'certified' live stop reproduces under replay at the live eps (S1)
  I5  replayed stops monotone in eps (S2); certified => no looser arm censored
  I6  measurement rows carry full replay context; 'certified' is a string
  I7  grading math sane
  I8  coverage events == N distinct records; spend <= budget + one search
  I9  (arm B) a wrong COUNT above the discoverable records NEVER yields a
      checksum certificate — the brake holds and the run ends uncertified
"""
import random

from tests.conftest import RoutedSearch, make_fake_llm, make_fetch
from webaggs.canon import normalize_name
from webaggs.pipeline import end_to_end
from webaggs.replay import replay_stop
from webaggs.storage import Ledger
from webaggs.truth import grade

EPS_ARMS = (0.05, 0.10, 0.20, 0.30)
BUDGET = 5.0


def _random_world(rng, player, adversarial=False):
    n_seasons = rng.randint(1, 8)
    start = rng.randint(1996, 2018)
    truth = {f"{y}-{(y + 1) % 100:02d}": rng.randint(5, 2500)
             for y in range(start, start + n_seasons)}
    seasons = list(truth)
    pages, n_pages = {}, rng.randint(1, 5)
    wrong_count = n_seasons + rng.randint(1, 3)
    for i in range(n_pages):
        covered = rng.sample(seasons, rng.randint(0, len(seasons)))
        rows = [(player if rng.random() < 0.8 else f"{player.split()[0][0]}. "
                 + player.split()[-1], s, truth[s], rng.uniform(0.82, 0.99))
                for s in covered]
        if rng.random() < 0.2:
            rows.append((f"Zork {player.split()[-1]}", rng.choice(seasons),
                         7777, 0.95))
        claims = []
        if adversarial:
            claims.append({"kind": "COUNT", "value": wrong_count})  # every page
        else:
            if rng.random() < 0.5:
                claims.append({"kind": "COUNT", "value": n_seasons})
            if rng.random() < 0.3:
                claims.append({"kind": "SUM", "value": sum(truth.values())})
        status = "paywall" if rng.random() < 0.15 else "ok"
        pages[f"https://w{i}.test/{player.split()[-1].lower()}"] = {
            "rows": rows, "claims": claims, "status": status}
    return truth, pages


def _check_run(trial, tmp_path, gate, player, truth, pages, adversarial):
    ledger = Ledger(tmp_path / f"r{trial}.sqlite")
    rid = f"r{trial}"
    ans = end_to_end(player, rid, ledger,
                     search=RoutedSearch(default=list(pages)), gate=gate,
                     llm=make_fake_llm(pages, player),
                     fetch=make_fetch(player, pages),
                     eps_g=0.05, delta_m=0.05, budget_usd=BUDGET, max_steps=25)

    stop = ledger.stop(rid)                                       # I1
    assert stop is not None and stop[1] in (
        "certified", "economic", "frontier_exhausted", "budget", "max_steps")

    assert ans.total == sum(r.value for r in ans.records)         # I2
    assert all("zork" not in r.record_key for r in ans.records)

    for r in ans.records:                                         # I3
        assert r.value == truth[r.season], (trial, r)

    ms = ledger.measurements(rid, metric="U_hat")
    for _s, _g, _m, _u, extra in ms:                              # I6
        assert isinstance(extra["certified"], str)
        for key in ("psi", "hot", "N", "claimed_count", "spent"):
            assert key in extra
    if ms:                                                        # I8
        last = ms[-1][4]
        assert last["N"] == len(ledger.coverage(rid)) == ans.n_records
        assert last["spent"] <= BUDGET + 0.02 + 1e-9

    if stop[1] == "certified":                                    # I4, I5
        rr = replay_stop(ms, 0.05)
        assert rr.stop_step == stop[0]
        for eps in EPS_ARMS:
            assert not replay_stop(ms, eps).censored
    prev = None
    for eps in EPS_ARMS:                                          # I5
        rr = replay_stop(ms, eps)
        if prev is not None:
            assert rr.stop_step is not None and rr.stop_step <= prev
        prev = rr.stop_step if rr.stop_step is not None else prev

    if adversarial:                                               # I9
        assert ans.certificate != "checksum", (trial, ans)
        assert stop[1] != "certified" or ans.certificate == "statistical"

    asserted = {r.season: r.value for r in ans.records}           # I7
    g = grade(normalize_name(player), truth, asserted)
    assert 0.0 <= g.record_recall <= 1.0 and g.n_spurious >= 0
    ledger.close()
    return ans


def test_invariants_over_benign_random_worlds(tmp_path, gate):
    rng = random.Random(7)
    for trial in range(40):
        player = f"Rando Trialson{trial}"
        truth, pages = _random_world(rng, player)
        _check_run(trial, tmp_path, gate, player, truth, pages,
                   adversarial=False)


def test_wrong_count_claims_never_certify(tmp_path, gate):
    """Arm B: every page corroborates a COUNT above the discoverable records.
    The cardinality brake must hold — no checksum certificate, ever."""
    rng = random.Random(41)
    outcomes = set()
    for trial in range(20):
        player = f"Adver Sarial{trial}"
        truth, pages = _random_world(rng, player, adversarial=True)
        ans = _check_run(1000 + trial, tmp_path, gate, player, truth, pages,
                         adversarial=True)
        outcomes.add(ans.certificate)
    assert "checksum" not in outcomes
    assert "abandoned" in outcomes               # the honest ending dominates
