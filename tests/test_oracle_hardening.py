"""OpenAlex oracle driver: pagination self-validation, schema traps,
transport hardening. All clients are fakes — offline."""
import pytest
from requests.exceptions import Timeout as ReqTimeout

from webaggs.oracle_openalex import OpenAlexDriver, OracleError


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


class _Client:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.seen_params = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        self.seen_params.append(params)
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _page(results, count, next_cursor):
    return _Resp({"meta": {"count": count, "next_cursor": next_cursor},
                  "results": results})


def _work(i, doi=True, title=None):
    return {"id": f"https://openalex.org/W{i}",
            "doi": f"https://doi.org/10.5555/x.{i}" if doi else None,
            "display_name": title if title is not None
            else f"Follow-up Study Number {i}",
            "publication_year": 2020}


def _drv(payloads):
    return OpenAlexDriver(pace_seconds=0, retry_cooldown=0,
                          client=_Client(payloads))


def test_cursor_walk_collects_all_pages_and_keys_doi_first():
    d = _drv([_page([_work(1), _work(2)], 3, "cur2"),
              _page([_work(3, doi=False)], 3, None)])
    rows = d.citers("W1", "2026-06-30")
    assert [r["key"] for r in rows] == [
        "doi:10.5555/x.1", "doi:10.5555/x.2", "t:follow up study number 3"]
    # the cutoff is part of the filter sent to the API
    assert "to_publication_date:2026-06-30" in \
        d.client.seen_params[0]["filter"]


def test_self_validation_refuses_partial_walk():
    d = _drv([_page([_work(1)], 5, None)])        # says 5, delivers 1
    with pytest.raises(OracleError, match="SELF-VALIDATION"):
        d.citers("W1", "2026-06-30")


def test_duplicate_keys_collapse_to_one_record():
    d = _drv([_page([_work(1), _work(1)], 2, None)])
    rows = d.citers("W1", "2026-06-30")
    assert len(rows) == 1


def test_unkeyable_citer_raises_never_shrinks_truth():
    bad = _work(9, doi=False, title="Hi")         # <3 words, no DOI
    d = _drv([_page([bad], 1, None)])
    with pytest.raises(OracleError, match="grading key"):
        d.citers("W1", "2026-06-30")


def test_missing_meta_count_raises():
    d = _drv([_Resp({"meta": {}, "results": []})])
    with pytest.raises(OracleError, match="meta.count"):
        d.citers("W1", "2026-06-30")


def test_work_meta_and_not_found():
    d = _drv([_Resp({"results": [{"id": "https://openalex.org/W7",
                                  "display_name": "T", "publication_year": 2017,
                                  "cited_by_count": 42}]})])
    m = d.work_meta("W7")
    assert m["cited_by_count"] == 42
    d = _drv([_Resp({"results": []})])
    with pytest.raises(OracleError, match="not found"):
        d.work_meta("W404")


def test_timeout_retried_once_then_succeeds_and_hard_404():
    ok = _page([_work(1)], 1, None)
    client = _Client([ReqTimeout("hang"), ok])
    d = OpenAlexDriver(pace_seconds=0, retry_cooldown=0, client=client)
    assert len(d.citers("W1", "2026-06-30")) == 1
    assert client.calls == 2
    d = _drv([_Resp({}, status=404)])
    with pytest.raises(OracleError, match="404"):
        d.work_meta("W0")
    assert d.client.calls == 1                    # 404 never retried
