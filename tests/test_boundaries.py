"""Boundary and edge-condition tests, citation domain."""
import pytest

from tests.conftest import TARGET, RoutedSearch, make_fake_llm, make_fetch
from webaggs.aggregate import aggregate_with_ci
from webaggs.corroborate import ResolvedRecord
from webaggs.pipeline import end_to_end
from webaggs.storage import Ledger
from webaggs.truth import grade, load_truth, write_truth


def _world(n):
    pages = {"https://a.test/citers": {"citations": [
        dict(title=f"Citing Paper Number {i} Study", doi=f"10.1234/n.{i}",
             cited_paper=TARGET, confidence=0.95) for i in range(n)]}}
    return pages


def _run(tmp_path, gate, pages, rid="b", **kw):
    ledger = Ledger(tmp_path / f"{rid}.sqlite")
    ans = end_to_end(TARGET, rid, ledger,
                     search=RoutedSearch(default=list(pages)), gate=gate,
                     llm=make_fake_llm(pages, TARGET),
                     fetch=make_fetch(TARGET, pages),
                     eps_g=0.30, delta_m=0.05, **kw)
    return ledger, ans


def test_novelty_counter_capped_but_all_records_enter(tmp_path, gate):
    ledger, ans = _run(tmp_path, gate, _world(20), budget_usd=5.0,
                       max_steps=10)
    assert ans.n_records == 20
    news = [e["new"] for _s, _g, _m, _u, e in ledger.measurements("b", "U_hat")]
    assert max(news) == 12                       # Y_CAP caps the counter only
    ledger.close()


def test_budget_and_max_steps_and_economic_stops(tmp_path, gate):
    ledger, ans = _run(tmp_path, gate, _world(3), rid="budget",
                       budget_usd=0.01, max_steps=10)
    assert ledger.stop("budget")[0] == 1 and ans.certificate == "abandoned"
    ledger.close()
    ledger, ans = _run(tmp_path, gate, _world(3), rid="cap",
                       budget_usd=5.0, max_steps=1)
    assert ledger.stop("cap")[1] == "max_steps"
    ledger.close()


def test_empty_search_results_leave_u_hat_at_one(tmp_path, gate):
    ledger, ans = _run(tmp_path, gate, {}, rid="empty", budget_usd=5.0,
                       max_steps=5)
    ms = ledger.measurements("empty", "U_hat")
    assert ms and all(u == 1.0 for _s, _g, _m, u, _e in ms)
    assert ans.certificate == "abandoned" and ans.n_records == 0
    ledger.close()


def test_irrelevant_page_feeds_the_phi_audit(tmp_path, gate):
    pages = _world(2)
    pages["https://soup.test/recipes"] = {"citations": []}

    def fetch(url, fid):
        src = make_fetch(TARGET, pages)(url, fid)
        if "soup" in url:
            src.main_text = "a long page about minestrone " * 30
        return src

    ledger = Ledger(tmp_path / "phi.sqlite")
    end_to_end(TARGET, "phi", ledger, search=RoutedSearch(default=list(pages)),
               gate=gate, llm=make_fake_llm(pages, TARGET), fetch=fetch,
               eps_g=0.30, delta_m=0.05, budget_usd=5.0, max_steps=10)
    rej = ledger.conn.execute("SELECT url FROM rejected_sources").fetchall()
    assert ("https://soup.test/recipes",) in rej
    ledger.close()


def test_truth_refuses_duplicates_and_grade_handles_edges(tmp_path):
    rows = [{"player": TARGET, "player_id": 1, "season": "doi:10.1/x",
             "pts": 1, "gp": 2020}] * 2
    with pytest.raises(RuntimeError, match="Duplicate"):
        write_truth(tmp_path, "dup", rows)
    ok = [{"player": TARGET, "player_id": 1, "season": "doi:10.1234/x.1",
           "pts": 1, "gp": 2020}]
    write_truth(tmp_path, "ok", ok)
    truth = load_truth(tmp_path, "ok")
    assert list(truth.values())[0] == {"doi:10.1234/x.1": 1}
    g = grade("p", {}, {"doi:10.1234/y.1": 1})
    assert g.record_recall == 0.0 and g.n_spurious == 1


def test_aggregate_interval_only_with_eps_f():
    recs = [ResolvedRecord("p|doi:10.1/x", "doi:10.1/x", 1, 1.0, 2)]
    ans = aggregate_with_ci("p", recs, eps_c=0.3, eps_f=None,
                            certificate="statistical", stop_reason="certified")
    assert ans.interval is None and ans.total == 1
