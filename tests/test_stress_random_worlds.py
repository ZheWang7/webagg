"""Randomized stress, citation domain: generated citer sets scattered over
random pages with duplicate DOI forms and contamination, run through the
REAL end_to_end. Claims are off (pre-registered), so endings are honest
(abandoned) except where a dense, fully re-covered world certifies
STATISTICALLY. Invariants per run:
  I1 stop recorded with a known reason
  I2 total == n_records (COUNT domain: every value is 1)
  I3 no contaminated key ever becomes a record
  I4 a certified live stop reproduces under replay at the live eps (S1)
  I5 replayed stops monotone in eps (S2)
  I6 measurement rows carry full replay context; certified is a string
  I7 grading sane; found keys are a subset of truth + known spurious forms
"""
import random

from tests.conftest import RoutedSearch, make_fake_llm, make_fetch
from webaggs.canon import normalize_name
from webaggs.pipeline import end_to_end
from webaggs.replay import replay_stop
from webaggs.storage import Ledger
from webaggs.truth import grade

EPS_ARMS = (0.20, 0.30, 0.40, 0.50)


def _world(rng, target):
    n = rng.randint(3, 60)
    truth = {}
    for i in range(n):
        if rng.random() < 0.7:
            truth[f"doi:10.9999/w.{i}"] = dict(
                title=f"Citing Work Number {i} Analysis",
                doi=f"10.9999/w.{i}")
        else:
            truth[f"t:citing work number {i} analysis"] = dict(
                title=f"Citing Work Number {i} Analysis", doi="")
    keys = list(truth)
    pages = {}
    for pg in range(rng.randint(1, 5)):
        subset = rng.sample(keys, rng.randint(0, len(keys)))
        cites = []
        for k in subset:
            v = truth[k]
            doi = v["doi"]
            if doi and rng.random() < 0.5:
                doi = f"https://doi.org/{doi.upper()}"   # URL/case variant
            cites.append(dict(title=v["title"], doi=doi,
                              cited_paper=target,
                              confidence=rng.uniform(0.82, 0.99)))
        if rng.random() < 0.25:                           # contamination
            cites.append(dict(title="Imposter Study On Something Else",
                              doi="10.4242/bad.1",
                              cited_paper="A Different Target Paper",
                              confidence=0.95))
        status = "paywall" if rng.random() < 0.15 else "ok"
        pages[f"https://w{pg}.test/{rng.randint(0, 10**6)}"] = {
            "citations": cites, "status": status}
    return truth, pages


def test_invariants_over_random_citation_worlds(tmp_path, gate):
    rng = random.Random(7)
    endings = set()
    for trial in range(30):
        target = f"Sample Landmark Paper Number {trial} Study"
        truth, pages = _world(rng, target)
        ledger = Ledger(tmp_path / f"r{trial}.sqlite")
        rid = f"r{trial}"
        ans = end_to_end(target, rid, ledger,
                         search=RoutedSearch(default=list(pages)), gate=gate,
                         llm=make_fake_llm(pages, target),
                         fetch=make_fetch(target, pages),
                         eps_g=0.30, delta_m=0.05, budget_usd=5.0,
                         max_steps=25)
        stop = ledger.stop(rid)
        assert stop and stop[1] in ("certified", "economic",
                                    "frontier_exhausted", "budget",
                                    "max_steps")                       # I1
        endings.add(ans.certificate)
        assert ans.total == ans.n_records == len(ans.records)         # I2
        assert all("bad.1" not in r.record_key for r in ans.records)  # I3
        ms = ledger.measurements(rid, metric="U_hat")
        for _s, _g, _m, _u, extra in ms:                              # I6
            assert isinstance(extra["certified"], str)
            for key in ("psi", "hot", "N", "claimed_count", "spent"):
                assert key in extra
        if stop[1] == "certified":                                    # I4
            assert replay_stop(ms, 0.30).stop_step == stop[0]
            assert ans.certificate == "statistical"
        prev = None                        # I5: ascending eps -> stop never
        for eps in sorted(EPS_ARMS):       # later, and once stopped, stays
            rr = replay_stop(ms, eps)
            if prev is not None:
                assert rr.stop_step is not None and rr.stop_step <= prev, \
                    (trial, eps)
            prev = rr.stop_step if rr.stop_step is not None else prev
        found = {r.record_key.split("|", 1)[1] for r in ans.records}
        assert found <= set(truth)                                    # I7
        g = grade(normalize_name(target), {k: 1 for k in truth},
                  {k: 1 for k in found})
        assert 0.0 <= g.record_recall <= 1.0 and g.n_spurious == 0
        ledger.close()
    assert "abandoned" in endings          # honest ending must occur
