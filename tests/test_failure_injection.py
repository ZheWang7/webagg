"""Failure injection: every external dependency misbehaving, and the
crash-resume path end to end. All network calls are mocked — nothing here
touches the real web."""
import json

import pytest

import webaggs.llm as llm_mod
import webaggs.search as search_mod
from tests.conftest import PAGES, PLAYER, RoutedSearch, make_fake_llm, \
    make_fetch
from webaggs.conformal import ConformalError, ConformalGate
from webaggs.fetch import fetch_url
from webaggs.llm import LLMError, call_llm, load_prompt
from webaggs.pipeline import end_to_end
from webaggs.replay import replay_stop
from webaggs.search import SerperBackend
from webaggs.storage import Ledger, fresh_ledger


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body if not isinstance(self._body, str) \
            else json.loads(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _ok_llm(content):
    return _Resp(200, {"choices": [{"message": {"content": content}}]})


# --- call_llm ---------------------------------------------------------------
def test_llm_transient_429_retries_once_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    replies = [_Resp(429, {}), _ok_llm('{"x": 1}')]
    sleeps = []
    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    assert call_llm("sys", "user") == {"x": 1}
    assert sleeps and not replies                 # paced once, both consumed


def test_llm_hard_500_twice_raises_loudly(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: _Resp(500, {}))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        call_llm("sys", "user")


def test_llm_malformed_json_retries_then_raises_with_raw_persisted(
        monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda *a, **k: _ok_llm("not json at all"))
    ledger = Ledger(tmp_path / "l.sqlite")
    with pytest.raises(LLMError):
        call_llm("sys", "user", ledger=ledger, run_id="r", purpose="extract")
    raws = ledger.conn.execute("SELECT raw FROM llm_calls").fetchall()
    ledger.close()
    assert len(raws) == 2 and all("not json" in r[0] for r in raws)


def test_llm_refuses_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError, match="not set"):
        call_llm("sys", "user")


def test_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist.txt")


# --- Serper -----------------------------------------------------------------
def test_search_refuses_without_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    with pytest.raises(Exception, match="not set"):
        SerperBackend()


def test_search_transient_429_then_parse(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    replies = [_Resp(429, {}), _Resp(200, {"organic": [
        {"link": "https://a.test/x", "title": "t"}, {"notalink": 1}]})]
    monkeypatch.setattr(search_mod.requests, "post",
                        lambda *a, **k: replies.pop(0))
    monkeypatch.setattr(search_mod.time, "sleep", lambda s: None)
    assert SerperBackend().search("q") == [{"url": "https://a.test/x",
                                            "title": "t"}]


def test_search_hard_failure_raises(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    monkeypatch.setattr(search_mod.requests, "post",
                        lambda *a, **k: _Resp(503, {}))
    monkeypatch.setattr(search_mod.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        SerperBackend().search("q")


# --- fetch_url real-HTTP branch (fake client) -------------------------------
class _Client:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc = resp, exc

    def get(self, url, headers=None, timeout=None):
        if self.exc:
            raise self.exc
        return self.resp


def test_fetch_outcome_classifier_branches():
    url = "https://site.test/x"
    assert fetch_url(url, "f", client=_Client(_Resp(403, ""))).status == "paywall"
    assert fetch_url(url, "f",
                     client=_Client(exc=TimeoutError())).status == "error"
    big = "<html><body>" + "stats " * 200 + "</body></html>"
    assert fetch_url(url, "f", client=_Client(_Resp(200, big))).status == "ok"
    stub = "<html><body>Subscribe to continue reading</body></html>"
    assert fetch_url(url, "f", client=_Client(_Resp(200, stub))).status == "paywall"
    tiny = "<html><body>hi</body></html>"
    assert fetch_url(url, "f", client=_Client(_Resp(200, tiny))).status == "empty"


# --- crash-resume end to end -------------------------------------------------
def test_crash_resume_policy_yields_one_clean_trajectory(tmp_path, gate):
    """A run crashing mid-player leaves a partial ledger; fresh_ledger moves
    it aside and the rerun produces ONE clean trajectory that replays."""
    calls = {"n": 0}
    good = make_fake_llm(PAGES, PLAYER)

    def crashing_llm(system, user, **kw):
        if kw.get("purpose") == "extract":
            calls["n"] += 1
            if calls["n"] == 2:
                raise LLMError("injected crash")
        return good(system, user, **kw)

    runs_dir = tmp_path / "runs"
    rid = "crashy"
    ledger = Ledger(runs_dir / f"{rid}.sqlite")
    with pytest.raises(LLMError):
        end_to_end(PLAYER, rid, ledger, search=RoutedSearch(default=list(PAGES)),
                   gate=gate, llm=crashing_llm, fetch=make_fetch(PLAYER, PAGES),
                   eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)
    ledger.close()

    ledger2 = fresh_ledger(runs_dir, rid)            # moves the partial aside
    assert (runs_dir / f"{rid}.crashed-1.sqlite").exists()
    ans = end_to_end(PLAYER, rid, ledger2, search=RoutedSearch(default=list(PAGES)),
                     gate=gate, llm=good, fetch=make_fetch(PLAYER, PAGES),
                     eps_g=0.05, delta_m=0.05, budget_usd=5.0, max_steps=30)
    ms = ledger2.measurements(rid, metric="U_hat")
    steps = [s for s, *_ in ms]
    assert steps == sorted(set(steps)), "duplicate/interleaved steps"
    live = ledger2.stop(rid)
    assert live[1] == "certified"
    assert replay_stop(ms, 0.05).stop_step == live[0]
    assert ans.total == 1450
    ledger2.close()


# --- conformal + ledger isolation + config gate ------------------------------
def test_conformal_load_success_and_malformed(tmp_path):
    good = tmp_path / "cal.csv"
    good.write_text("confidence,correct\n" +
                    "".join(f"0.{80 + i % 19:02d},1\n" for i in range(25)))
    gate = ConformalGate.load(good)
    assert 0.0 < gate.threshold < 1.0 and gate.n_cal == 25
    bad = tmp_path / "bad.csv"
    bad.write_text("conf,ok\n0.9,1\n")
    with pytest.raises(ConformalError, match="Malformed calibration"):
        ConformalGate.load(bad)


def test_ledger_isolates_run_ids(tmp_path):
    led = Ledger(tmp_path / "multi.sqlite")
    led.log_measurement("r1", 1, "p", "U_hat", 0.5, {"psi": 1})
    led.log_measurement("r2", 1, "p", "U_hat", 0.9, {"psi": 2})
    led.log_stop("r1", 3, "certified")
    assert [v for _s, _g, _m, v, _e in led.measurements("r1")] == [0.5]
    assert led.stop("r2") is None and led.stop("r1")[1] == "certified"
    led.close()


def test_live_gate_env(monkeypatch):
    from webaggs import config
    monkeypatch.setenv("RUN_LIVE", "1")
    assert config.live_enabled()
    monkeypatch.setenv("RUN_LIVE", "0")
    assert not config.live_enabled()


def test_second_crash_gets_its_own_sidecar(tmp_path):
    from webaggs.storage import Ledger, fresh_ledger
    runs = tmp_path / "runs"
    Ledger(runs / "r.sqlite").close()
    fresh_ledger(runs, "r").close()                 # -> crashed-1
    Ledger(runs / "r.sqlite").close()
    led = fresh_ledger(runs, "r")                   # -> crashed-2, not clobber
    led.add_resolved("r", "p|2019-20", "2019-20", 10, 1.0, 2)
    assert led.resolved("r") == [("p|2019-20", "2019-20", 10, 1.0, 2)]
    led.close()
    assert (runs / "r.crashed-1.sqlite").exists()
    assert (runs / "r.crashed-2.sqlite").exists()
