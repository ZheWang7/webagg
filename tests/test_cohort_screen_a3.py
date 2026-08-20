"""Attempt #3 instrument batch -- offline tests.

  test_snapshot_alignment            -> tolerance matching may use chain
                                        snapshots; error still vs FINAL
  test_ambiguous_truth_pairs         -> close truth amounts are surfaced
  test_snapshots_roundtrip           -> formd emits + reloads snapshots
  test_norm_and_score                -> name matching: suffixes stripped,
                                        containment high, unrelated low
  test_parse_browse_atom_multi       -> multi-match Atom -> (cik, name)
  test_formd_stats                   -> submissions -> counts + date span
  test_read_candidates_skips_comments
  test_screen_one_offline            -> MockTransport end-to-end row
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webagg.cohort_screen import (formd_stats, name_score, norm_name,  # noqa: E402
                                  parse_browse_atom, parse_browse_html,
                                  read_candidates, screen_one)
from webagg.formd import build_truth_entity, parse_form_d  # noqa: E402
from webagg.formd import load_truth_entity, save_truth_entity  # noqa: E402
from webagg.risk_control import (TruthEntity, TruthRecord,  # noqa: E402
                                 ambiguous_truth_pairs, fidelity_loss,
                                 match_to_truth)


def _rec(amount, kind="funding_round/unknown"):
    from types import SimpleNamespace
    return {"entity_id": "e", "record_kind": kind,
            "attributes": {"amount": SimpleNamespace(value=str(amount),
                                                     value_num=float(amount))}}


def test_snapshot_alignment():
    # chain grew 200M -> 349M; press holds the 200M announcement tranche
    t = TruthEntity("g", (TruthRecord("funding_round|2018-02-07",
                                      349_000_000.0, "2018-02-07",
                                      amount_snapshots=(200_000_000.0,)),))
    rec = _rec(200_000_000)
    # final-only distance is 43% -> no match without snapshots
    aligned = match_to_truth([rec], t, amount_tol=0.05)
    assert aligned[0][1] is not None          # snapshot got it aligned
    # and the ERROR is still graded against the FINAL amount (149M short)
    loss = fidelity_loss([rec], t, amount_tol=0.05)
    assert abs(loss - 149_000_000 / 349_000_000) < 1e-6


def test_ambiguous_truth_pairs():
    t = TruthEntity("g", (
        TruthRecord("funding_round|2014-12-15", 220_164_327.0),
        TruthRecord("funding_round|2020-06-15", 224_999_926.0),   # 2.2% apart
        TruthRecord("funding_round|2017-03-02", 413_018_774.0)))
    pairs = ambiguous_truth_pairs(t, amount_tol=0.05)   # 2*tol = 10%
    assert len(pairs) == 1
    assert {pairs[0][0], pairs[0][1]} == {"funding_round|2014-12-15",
                                          "funding_round|2020-06-15"}


def test_snapshots_roundtrip(tmp_path):
    xml = lambda amt, amend, prev: f"""<?xml version="1.0"?><edgarSubmission>
      <primaryIssuer><cik>9</cik><entityName>Acme</entityName></primaryIssuer>
      <offeringData><typeOfFiling><newOrAmendment>
        {'<isAmendment>true</isAmendment><previousAccessionNumber>' + prev + '</previousAccessionNumber>' if amend else '<isAmendment>false</isAmendment>'}
      </newOrAmendment>
      <dateOfFirstSale><value>2018-02-07</value></dateOfFirstSale>
      </typeOfFiling><offeringSalesAmounts>
        <totalOfferingAmount>400000000</totalOfferingAmount>
        <totalAmountSold>{amt}</totalAmountSold>
      </offeringSalesAmounts></offeringData></edgarSubmission>"""
    d = parse_form_d(xml("200000000", False, ""),
                     accession="0000000009-18-000001", filing_date="2018-02-09")
    da = parse_form_d(xml("349000000", True, "0000000009-18-000001"),
                      accession="0000000009-18-000002", filing_date="2018-04-20")
    truth, meta = build_truth_entity("cik0000000009", [d, da])
    assert truth.records[0].amount == 349_000_000.0        # final wins
    assert truth.records[0].amount_snapshots == (200_000_000.0,)
    p = save_truth_entity(tmp_path, meta)
    truth2, _ = load_truth_entity(p)
    assert truth2 == truth                                 # snapshots survive


def test_norm_and_score():
    assert norm_name("Databricks, Inc.") == "databricks"
    assert name_score("Databricks", "DATABRICKS INC") >= 0.95
    assert name_score("Figure AI", "FIGURE AI INC") >= 0.95
    assert name_score("Glow Security", "GLOWING GARDENS LLC") < 0.60
    # Strict-subset containment lands in the weak_match band (0.60-0.85):
    # a press name can't tell the company from a same-named fund vehicle,
    # so the CIK needs a human eye (attempt-#3 instrument, pre-registered
    # before any human_verdict existed).
    assert 0.60 <= name_score("Robinhood", "ROBINHOOD CASINO LLC") < 0.85
    assert 0.60 <= name_score("Ramp", "RAMP BUSINESS CORP") < 0.85
    # difflib near-miss on a DIFFERENT company: must stay under min_ok=0.90
    # so a human confirms the CIK (AFFIRMXH is not Affirm).
    assert 0.60 <= name_score("Affirm", "AFFIRMXH INC.") < 0.90


def test_parse_browse_atom_multi():
    atom = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>DATABRICKS INC (CIK 0001327688)</title>
        <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=1327688&amp;type=D"/>
      </entry>
      <entry><title>DATABRICKS HOLDINGS LLC</title>
        <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=9999999&amp;type=D"/>
      </entry>
    </feed>"""
    out = parse_browse_atom(atom)
    assert ("0001327688", "DATABRICKS INC (CIK 0001327688)") == out[0]
    assert len(out) == 2


def test_formd_stats():
    subs = {"filings": {"recent": {
        "form": ["D", "10-K", "D/A", "D"],
        "filingDate": ["2019-05-01", "2020-01-01", "2019-08-01",
                       "2021-03-03"]}}}
    st = formd_stats(subs)
    assert st == {"n_formd": 3, "n_original_d": 2,
                  "first_formd": "2019-05-01", "last_formd": "2021-03-03"}


def test_read_candidates_skips_comments(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("# tier A\nStripe\n\n# tier B\nRamp\n")
    assert read_candidates(f) == ["Stripe", "Ramp"]


def test_screen_one_offline():
    atom = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>RAMP BUSINESS CORP</title>
        <link href="https://x?CIK=1799567&amp;t=D"/></entry></feed>"""
    subs = {"filings": {"recent": {"form": ["D", "D/A"],
                                   "filingDate": ["2020-02-01",
                                                  "2020-06-01"]}}}

    def handler(request: httpx.Request) -> httpx.Response:
        if "browse-edgar" in str(request.url):
            return httpx.Response(200, text=atom)
        return httpx.Response(200, json=subs)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    row = screen_one("Ramp", client, pause_s=0.0)
    # weak_match, not ok: "BUSINESS" is an extra distinctive token, so the
    # CIK gets flagged for human confirmation (subset-demotion rule).
    assert row["verdict"] == "weak_match" and row["cik"] == "0001799567"
    assert row["n_formd"] == "2" and row["first_formd"] == "2020-02-01"


# --- SEC Atom regression (2026-08) + HTML fallback --------------------------

# The broken multi-match shape actually served by SEC: title is an entry
# ATTRIBUTE holding a Perl array artifact, no <title> child, no
# conformed-name element -- only the CIK survives.
_BROKEN_ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry title="ARRAY(0x55d7321182b8)">
    <content type="text/xml">
      <company-info name="ARRAY(0x55d7321232e8)">
        <cik>0001786927</cik>
      </company-info>
    </content>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001786927&amp;type=D" type="text/html"/>
  </entry>
</feed>"""

_HTML_MULTI = """<html><body><table class="tableFile2">
<tr><td><a href="/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001786927&amp;type=D">0001786927</a></td>
<td>Robinhood Casino LLC</td><td>NY</td></tr>
<tr><td><a href="/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001783879&amp;type=D">0001783879</a></td>
<td>Robinhood Markets, Inc.</td><td>CA</td></tr>
</table></body></html>"""

_HTML_SINGLE = """<html><body>
<span class="companyName">Snowflake Inc.
<acronym title="Central Index Key">CIK</acronym>#:
<a href="/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001640147">0001640147</a></span>
</body></html>"""


def test_parse_browse_atom_broken_yields_nothing():
    # No names anywhere -> nothing to match on; the caller must fall back.
    assert parse_browse_atom(_BROKEN_ATOM) == []


def test_parse_browse_html_multi_and_single():
    out = parse_browse_html(_HTML_MULTI)
    assert ("0001786927", "Robinhood Casino LLC") in out
    assert ("0001783879", "Robinhood Markets, Inc.") in out
    assert parse_browse_html(_HTML_SINGLE) == [("0001640147",
                                                "Snowflake Inc.")]


def test_screen_one_falls_back_to_html():
    """Broken atom -> one HTML request -> best match resolved from HTML."""
    subs = {"filings": {"recent": {"form": ["D"],
                                   "filingDate": ["2019-05-01"]}}}
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "browse-edgar" in url and "output=atom" in url:
            return httpx.Response(200, text=_BROKEN_ATOM)
        if "browse-edgar" in url:                      # HTML fallback
            return httpx.Response(200, text=_HTML_MULTI)
        return httpx.Response(200, json=subs)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    row = screen_one("Robinhood", client, pause_s=0.0)
    # Both same-scored (0.75) candidates are subset matches; either way the
    # verdict is weak_match with a real CIK attached -- the human decides.
    assert row["verdict"] == "weak_match" and row["cik"]
    assert row["n_formd"] == "1"
    assert row["note"].startswith("alts: ")   # runner-up visible to human
    assert sum("output=atom" in u for u in calls) == 1  # atom tried once
    assert any("browse-edgar" in u and "output=atom" not in u
               for u in calls)                          # fallback happened


def test_merged_submissions_walks_older_pages():
    """The Coinbase failure: 0 Form Ds in recent, all on the older page.
    Stats must come from the MERGED filing list, and first_formd must
    reflect the older page's dates (the Snowflake failure)."""
    from webagg.cohort_screen import merged_submissions
    subs = {"filings": {
        "recent": {"form": ["10-K", "8-K", "D"],
                   "filingDate": ["2024-01-01", "2023-06-01", "2022-04-20"]},
        "files": [{"name": "CIK0000000001-submissions-001.json",
                   "filingCount": 2}]}}
    older = {"form": ["D", "D/A"],
             "filingDate": ["2017-04-11", "2018-01-01"]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert "submissions-001" in str(request.url)
        return httpx.Response(200, json=older)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    st = formd_stats(merged_submissions(subs, client, pause_s=0.0))
    assert st["n_formd"] == 3                       # 1 recent + 2 older
    assert st["first_formd"] == "2017-04-11"        # from the older page
    assert st["last_formd"] == "2022-04-20"


def test_get_retry_survives_timeouts():
    from webagg.cohort_screen import _get_retry
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] < 3:
            raise httpx.ReadTimeout("slow SEC")        # first two attempts
        return httpx.Response(200, text="ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = _get_retry(client, "https://x", tries=3, backoff_s=0.0)
    assert r.status_code == 200 and state["n"] == 3
