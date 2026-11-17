"""The index plug-in against FAKE transports shaped like the real APIs
(OpenAlex works, Europe PMC search, Semantic Scholar graph). Offline.
The live shape of each API is checked by scripts/probe_enumerators.py and
the online sanity tests — these tests pin OUR side of the contract."""
import pytest

from webaggs import design
from webaggs.indexes import (EuropePMCIndex, IndexUnavailable, OpenAlexIndex,
                             SemanticScholarIndex, Target, make_indexes)

T = Target(title="A Landmark Paper On Things", doi="10.1234/land.1",
           pmid="555")


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code, self._p = status, payload

    def json(self):
        if self._p is None:
            raise ValueError("not json")
        return self._p


class _Client:
    """Replays queued responses; records what was asked."""

    def __init__(self, *responses):
        self.q, self.seen = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.seen.append((url, dict(params or {}), dict(headers or {})))
        return self.q.pop(0)


def _ix(cls, *responses, **kw):
    return cls(client=_Client(*responses), pace_seconds=0, retries=2,
               cooldown=0, **kw)


# --- OpenAlex ---------------------------------------------------------------
def test_openalex_resolves_pages_and_never_logs_the_key(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRET-KEY")
    ix = _ix(OpenAlexIndex,
             _Resp({"results": [{"id": "https://openalex.org/W77"}]}),
             _Resp({"meta": {"count": 3, "next_cursor": "c2"},
                    "results": [
                        {"doi": "https://doi.org/10.1/AAAA.1",
                         "display_name": "First Citing Paper"},
                        {"doi": None, "display_name": "No Doi Paper"}]}))
    handle = ix.resolve(T)
    assert handle == "W77"
    src, page = ix.fetch_page(T, handle, "*", 50, "2023-12-31", "fm1")
    assert src.status == "ok" and src.source_class == "citation_index"
    assert src.domain == "api.openalex.org"
    assert "SECRET-KEY" not in src.url              # ledger URL is keyless
    sent = ix.client.seen[1][1]
    assert sent["api_key"] == "SECRET-KEY"          # ... but the key was sent
    assert sent["filter"] == ("cites:W77,to_publication_date:2023-12-31,"
                              "has_doi:true")
    assert sent["per-page"] == 50 and sent["cursor"] == "*"
    assert page.next_cursor == "c2" and page.total == 3
    assert [r["doi"] for r in page.rows] == ["https://doi.org/10.1/AAAA.1", ""]
    # rendered text names the target (relevance gate + ER guard read it)
    assert "Papers citing: A Landmark Paper On Things" in src.main_text
    assert "First Citing Paper || doi: https://doi.org/10.1/AAAA.1" \
        in src.main_text


def test_openalex_end_of_walk_and_unresolvable_target():
    ix = _ix(OpenAlexIndex, _Resp({"meta": {"count": 3, "next_cursor": None},
                                   "results": []}))
    src, page = ix.fetch_page(T, "W77", "c9", 50, "2023-12-31", "fm")
    assert src.status == "empty" and page.next_cursor is None
    ix = _ix(OpenAlexIndex, _Resp({"results": []}))
    with pytest.raises(IndexUnavailable):
        ix.resolve(T)
    with pytest.raises(IndexUnavailable):
        _ix(OpenAlexIndex).resolve(Target(title="x", doi="not-a-doi"))


# --- Europe PMC -------------------------------------------------------------
def test_europepmc_uses_cites_search_with_cutoff_and_cursor():
    ix = _ix(EuropePMCIndex,
             _Resp({"hitCount": 120, "nextCursorMark": "AoE2",
                    "resultList": {"result": [
                        {"doi": "10.1/bbbb.2", "title": "A Citing Study",
                         "pmid": "9"},
                        {"title": "Entry Without Doi", "pmid": "10"}]}}))
    assert ix.resolve(T) == "555"                   # PMID given: no lookup
    src, page = ix.fetch_page(T, "555", "*", 50, "2023-12-31", "fm")
    q = ix.client.seen[0][1]
    assert q["query"] == ("CITES:555_MED AND "
                          "FIRST_PDATE:[1900-01-01 TO 2023-12-31]")
    assert q["cursorMark"] == "*" and q["pageSize"] == 50
    assert src.status == "ok" and page.total == 120
    assert page.next_cursor == "AoE2"
    assert [r["doi"] for r in page.rows] == ["10.1/bbbb.2", ""]


def test_europepmc_walk_ends_when_the_cursor_repeats_or_rows_run_out():
    ix = _ix(EuropePMCIndex, _Resp({"hitCount": 2, "nextCursorMark": "SAME",
                                    "resultList": {"result": [
                                        {"doi": "10.1/cccc.3", "title": "t"}]}}))
    _src, page = ix.fetch_page(T, "555", "SAME", 50, "2023-12-31", "fm")
    assert page.next_cursor is None                 # Solr repeats the mark
    ix = _ix(EuropePMCIndex, _Resp({"hitCount": 2, "nextCursorMark": "X",
                                    "resultList": {"result": []}}))
    src, page = ix.fetch_page(T, "555", "W", 50, "2023-12-31", "fm")
    assert src.status == "empty" and page.next_cursor is None


def test_europepmc_resolves_pmid_from_doi_or_reports_unavailable():
    ix = _ix(EuropePMCIndex,
             _Resp({"resultList": {"result": [{"pmid": "4242"}]}}))
    assert ix.resolve(Target(title="x", doi="10.1234/land.1")) == "4242"
    ix = _ix(EuropePMCIndex, _Resp({"resultList": {"result": []}}))
    with pytest.raises(IndexUnavailable, match="PMID"):
        ix.resolve(Target(title="x", doi="10.1234/land.1"))


# --- Semantic Scholar (optional third) --------------------------------------
def test_semantic_scholar_filters_after_cutoff_but_keeps_walking(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "S2KEY")
    ix = _ix(SemanticScholarIndex,
             _Resp({"offset": 0, "next": 2, "data": [
                 {"citingPaper": {"title": "Too New", "year": 2025,
                                  "externalIds": {"DOI": "10.1/new.1"}}},
                 {"citingPaper": {"title": "Also New", "year": 2024,
                                  "externalIds": {"DOI": "10.1/new.2"}}}]}),
             _Resp({"offset": 2, "data": [
                 {"citingPaper": {"title": "In Window", "year": 2022,
                                  "externalIds": {"DOI": "10.1/old.1"}}}]}))
    handle = ix.resolve(T)
    assert handle == "DOI:10.1234/land.1"
    src, page = ix.fetch_page(T, handle, "0", 2, "2023-12-31", "fm")
    # every row was after the cutoff: an EMPTY page, but NOT the end
    assert src.status == "empty" and page.next_cursor == "2"
    assert ix.client.seen[0][2]["x-api-key"] == "S2KEY"   # key as a header
    assert "S2KEY" not in src.url
    src, page = ix.fetch_page(T, handle, "2", 2, "2023-12-31", "fm")
    assert src.status == "ok" and page.next_cursor is None
    assert [r["doi"] for r in page.rows] == ["10.1/old.1"]
    monkeypatch.delenv("S2_API_KEY")
    with pytest.raises(IndexUnavailable, match="S2_API_KEY"):
        _ix(SemanticScholarIndex).resolve(T)


# --- shared machinery -------------------------------------------------------
def test_transient_errors_retry_then_surface_as_error_not_data():
    ok = _Resp({"hitCount": 1, "resultList": {"result": [
        {"doi": "10.1/dddd.4", "title": "t"}]}})
    ix = _ix(EuropePMCIndex, _Resp(None, status=503), ok)
    src, page = ix.fetch_page(T, "555", "*", 50, "2023-12-31", "fm")
    assert src.status == "ok" and len(ix.client.seen) == 2   # retried once
    ix = _ix(EuropePMCIndex, _Resp(None, status=503), _Resp(None, status=503))
    src, page = ix.fetch_page(T, "555", "*", 50, "2023-12-31", "fm")
    assert src.status == "error" and page is None   # never an empty "ok"
    ix = _ix(OpenAlexIndex, _Resp(None, status=403))
    src, page = ix.fetch_page(T, "W1", "*", 50, "2023-12-31", "fm")
    assert src.status == "paywall" and page is None  # bad key = censored
    ix = _ix(OpenAlexIndex, _Resp({"unexpected": "shape", "results": "x"}))
    src, page = ix.fetch_page(T, "W1", "*", 50, "2023-12-31", "fm")
    assert src.status == "error" and page is None    # schema surprise


def test_page_size_bound_is_enforced_even_if_the_api_over_delivers():
    many = [{"doi": f"10.1/over.{i}", "title": f"Paper {i}"} for i in range(9)]
    ix = _ix(EuropePMCIndex, _Resp({"hitCount": 9, "nextCursorMark": "n",
                                    "resultList": {"result": many}}))
    _src, page = ix.fetch_page(T, "555", "*", 5, "2023-12-31", "fm")
    assert len(page.rows) == 5                       # Y_CAP holds regardless


def test_make_indexes_builds_only_known_enumerators_never_the_oracle():
    assert [i.name for i in make_indexes()] == ["openalex", "europepmc"]
    with pytest.raises(ValueError):
        make_indexes(("opencitations",))             # the oracle is not here


# --- design arithmetic ------------------------------------------------------
def test_design_arithmetic_matches_the_stop_test_formula():
    from webaggs.frontier import FrontierState
    st = FrontierState()
    st.Y_CAP = 50
    for i in range(1000):
        st.covered[f"p|{i}"].add("f")
        st.record_stratum[f"p|{i}"] = "p"
    assert design.psi_at(1000, 50, 600) == pytest.approx(
        st.psi({"p"}, 0.05, 1.0, 600))               # same formula as live
    # n_needed is exactly where psi crosses the room eps - u
    n = design.n_needed(0.5, 50, 600, u=0.1)
    assert design.psi_at(int(n) + 1, 50, 600) < 0.4 < design.psi_at(
        int(n) - 1, 50, 600)
    assert design.n_needed(0.5, 50, 600, u=0.5) == float("inf")


def test_feasible_band_and_min_tau():
    assert design.feasible_band(0.5, 50, 200) is None    # tau too small
    lo, hi = design.feasible_band(0.5, 50, 1000, u=0.0)
    assert lo < hi
    # the band's N range scales with the page size; min tau does not
    lo2, hi2 = design.feasible_band(0.5, 100, 1000, u=0.0)
    assert lo2 == pytest.approx(2 * lo) and hi2 == pytest.approx(2 * hi)
    # more singletons need a larger budget; at u >= eps nothing works
    assert design.min_tau(0.5, u=0.0) < design.min_tau(0.5, u=0.2)
    with pytest.raises(ValueError):
        design.min_tau(0.5, u=0.5)
