"""OpenCitations oracle driver: universe rules, cutoff, self-validation,
transport hardening. All clients are fakes — offline."""
import pytest
from requests.exceptions import Timeout as ReqTimeout

from webaggs.oracle_opencitations import (OpenCitationsDriver, OracleError,
                                          citing_dois)


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._p = status, payload

    def json(self):
        if isinstance(self._p, Exception):
            raise self._p
        return self._p


class _Client:
    """Routes by endpoint: citations/... -> rows, citation-count/... -> count."""

    def __init__(self, rows, count=None, fail_first=None):
        self.rows = rows
        self.count = len(rows) if count is None else count
        self.fail_first = fail_first
        self.urls, self.headers = [], []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        self.headers.append(headers or {})
        if self.fail_first is not None:
            item, self.fail_first = self.fail_first, None
            if isinstance(item, Exception):
                raise item
            return item
        if "/citation-count/" in url:
            return _Resp([{"count": str(self.count)}])
        return _Resp(self.rows)


def _row(doi="10.5555/x.1", creation="2020-05-01", extra=""):
    citing = f"omid:br/061 {('doi:' + doi) if doi else ''} {extra}".strip()
    return {"oci": "1-2", "citing": citing, "cited": "doi:10.1234/target",
            "creation": creation, "timespan": "P1Y"}


def _drv(rows, **kw):
    return OpenCitationsDriver(pace_seconds=0, retry_cooldown=0,
                               client=_Client(rows, **kw))


def test_citing_field_parsing():
    assert citing_dois("omid:br/1 doi:10.1234/ABC.5 pmid:99") == ["10.1234/abc.5"]
    assert citing_dois("omid:br/1 pmid:99") == []
    assert citing_dois("doi:10.1234/a doi:10.1234/b") == ["10.1234/a",
                                                          "10.1234/b"]
    assert citing_dois(None) == []


def test_universe_and_cutoff_rules_with_a_full_report():
    d = _drv([
        _row("10.5555/x.1", "2020-05-01"),             # in
        _row("10.5555/X.2", "2023"),                   # year precision: in
        _row("10.5555/x.3", "2023-12"),                # month precision: in
        _row("10.5555/x.4", "2024-01-01"),             # after cutoff: out
        _row(None, "2019-01-01", extra="pmid:7"),      # no DOI: out
        _row("10.5555/x.6", ""),                       # undated: out
        _row("10.5555/x.1", "2021-01-01"),             # duplicate DOI: out
        _row("10.5555/x.8", "2022", extra="doi:10.5555/alias.8"),  # multi-DOI
    ])
    rows, rep = d.citers("https://doi.org/10.1234/TARGET", "2023-12-31")
    assert [r["key"] for r in rows] == [
        "doi:10.5555/x.1", "doi:10.5555/x.2", "doi:10.5555/x.3",
        "doi:10.5555/x.8"]
    assert rep == {"raw_rows": 8, "no_doi": 1, "no_date": 1,
                   "after_cutoff": 1, "duplicate": 1, "multi_doi": 1,
                   "truth_rows": 4}
    # the target DOI was canonicalized into the request path
    assert d.client.urls[0].endswith("/citations/doi:10.1234/target")


def test_cutoff_must_be_a_year_end():
    for bad in ("2023-06-30", "2023", "", None):
        with pytest.raises(OracleError, match="31 December"):
            _drv([_row()]).citers("10.1234/target", bad)


def test_self_validation_refuses_a_truncated_list():
    d = _drv([_row()], count=5)                    # count says 5, list has 1
    with pytest.raises(OracleError, match="SELF-VALIDATION"):
        d.citers("10.1234/target", "2023-12-31")


def test_transport_hardening():
    # one transient timeout is retried
    d = _drv([_row()], fail_first=ReqTimeout())
    rows, _rep = d.citers("10.1234/target", "2023-12-31")
    assert len(rows) == 1
    # a non-list payload (error object) is refused, not treated as empty
    d = OpenCitationsDriver(pace_seconds=0, retry_cooldown=0,
                            client=_Client({"error": "boom"}))
    with pytest.raises(OracleError, match="payload shape"):
        d.citers("10.1234/target", "2023-12-31")
    with pytest.raises(OracleError, match="Not a DOI"):
        _drv([_row()]).citers("nonsense", "2023-12-31")


def test_optional_access_token_is_sent_as_a_header(monkeypatch):
    monkeypatch.setenv("OPENCITATIONS_ACCESS_TOKEN", "tok-123")
    d = _drv([_row()])
    d.citation_count("10.1234/target")
    assert d.client.headers[0]["authorization"] == "tok-123"
