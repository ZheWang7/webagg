"""Smoke test for guide ch. 5 (llm.py / search.py / fetch.py updates).

Run:  pytest tests/test_smoke_ch5.py -v
All provider/network calls are mocked -- no API keys needed.
"""
import json
import types
import httpx
import pytest

from webagg import llm, fetch, search
from webagg.storage import get_session, MeasurementRow


# ---------------------------------------------------------------- llm.py
class _FakeResp:
    """Minimal stand-in for an OpenAI chat.completions response."""
    def __init__(self, text, finish="stop", in_tok=100, out_tok=20):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=text), finish_reason=finish)]
        self.usage = types.SimpleNamespace(prompt_tokens=in_tok,
                                           completion_tokens=out_tok)


def test_reprompt_once_and_cost_logging(tmp_path, monkeypatch):
    """Bad JSON on the first reply -> ONE corrective re-prompt (guide ch. 5),
    and the call lands in measurements with summed token counts."""
    replies = [_FakeResp("here you go: {broken"),        # invalid JSON
               _FakeResp(json.dumps({"relevant": True}))]  # corrected
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return replies[len(calls) - 1]

    monkeypatch.setattr(llm._client_openai.chat.completions, "create", fake_create)

    session = get_session(str(tmp_path / "ch5.sqlite"))
    llm.set_llm_logger(session, "run_ch5")
    llm.set_llm_step(7)
    try:
        out = llm.call_llm(system="sys", user="usr", purpose="relevance")
    finally:
        llm.set_llm_logger(None, "")       # don't leak the sink to other tests

    assert out["payload"] == {"relevant": True}
    assert len(calls) == 2                                  # exactly one re-prompt
    # the corrective message shows the model its own bad output
    assert calls[1]["messages"][-2]["content"].startswith("here you go")
    assert out["input_tokens"] == 200 and out["output_tokens"] == 40  # both calls billed

    session.commit()
    row = session.query(MeasurementRow).filter_by(metric="llm_call").one()
    assert row.step == 7 and row.value == 240.0             # total tokens
    assert row.extra["purpose"] == "relevance" and row.extra["reprompted"] is True


def test_default_model_is_cheap_config_key(monkeypatch):
    from webagg import config
    seen = {}

    def fake_create(**kwargs):
        seen["model"] = kwargs["model"]
        return _FakeResp(json.dumps({"ok": 1}))

    monkeypatch.setattr(llm._client_openai.chat.completions, "create", fake_create)
    llm.call_llm(system="s", user="u")
    assert seen["model"] == config.MODEL_CHEAP              # guide ch. 5: config key


# -------------------------------------------------------------- fetch.py
_HTML = "<html><body>" + "Acme filed a Form D. " * 30 + "</body></html>"


def _fake_response(url):
    return httpx.Response(200, headers={"content-type": "text/html"},
                          text=_HTML, request=httpx.Request("GET", url))


def test_fetch_cache_one_url_one_source(monkeypatch):
    """Same URL from two formulations -> ONE fetch, ONE Source (one
    source_id). Refetching would mint a second source_id and the same page
    would masquerade as two witnesses in corroboration."""
    fetch.clear_fetch_cache()
    n_gets = []
    monkeypatch.setattr(fetch, "_get",
                        lambda url: (n_gets.append(url), _fake_response(url))[1])

    a = fetch.fetch_url("https://www.sec.gov/formd/acme", formulation_id="f1")
    b = fetch.fetch_url("https://www.sec.gov/formd/acme", formulation_id="f2")
    assert len(n_gets) == 1                                 # cached, not refetched
    assert a is b and a.formulation_id == "f1"              # first-discoverer attribution
    # §4.1 fields populated at birth:
    assert a.source_class == "regulatory"
    assert a.identity_anchored is True                      # registry -> qbar-cap exempt
    fetch.clear_fetch_cache()


def test_fetch_negative_cache(monkeypatch):
    """Dead URLs are cached as misses: not re-tried by every formulation."""
    fetch.clear_fetch_cache()
    n_gets = []

    def dead(url):
        n_gets.append(url)
        return httpx.Response(404, headers={"content-type": "text/html"},
                              text="", request=httpx.Request("GET", url))

    monkeypatch.setattr(fetch, "_get", dead)
    assert fetch.fetch_url("https://x.com/gone", "f1") is None
    assert fetch.fetch_url("https://x.com/gone", "f2") is None
    assert len(n_gets) == 1
    fetch.clear_fetch_cache()


def test_publish_time_dateparser():
    """Messy human date string -> UTC-naive datetime (project convention)."""
    meta = types.SimpleNamespace(date="January 5, 2026", title="t")
    dt = fetch._parse_publish_time(meta)
    assert (dt.year, dt.month, dt.day) == (2026, 1, 5)
    assert dt.tzinfo is None                                # UTC-naive convention
    assert fetch._parse_publish_time(None) is None


# -------------------------------------------------------------- search.py
def test_search_tags_formulation(monkeypatch):
    """Every result carries the formulation that surfaced it (guide ch. 5)."""
    fake = httpx.Response(
        200, json={"organic": [{"link": "https://a.com", "title": "A",
                                "snippet": "s"},
                               {"link": "https://b.com"}]},
        request=httpx.Request("POST", "https://google.serper.dev/search"))
    monkeypatch.setattr(search.httpx, "post", lambda *a, **k: fake)
    monkeypatch.setenv("SERPER_API_KEY", "dummy")

    results = search.SerperBackend().search("acme funding", k=2,
                                            formulation_id="f_042")
    assert [r["formulation_id"] for r in results] == ["f_042", "f_042"]
    assert results[1]["title"] == "" and results[1]["snippet"] == ""
