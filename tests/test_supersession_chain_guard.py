"""Regression test for the chain-guard fixes found by Experiment 4 on live
EDGAR data (notebooks/exp4_echo_robustness.ipynb).

The failure it locks out: an amendment (Form D/A) shares long verbatim
boilerplate with the Form D it supersedes, so the copy detector gives the
successor derivation edges from its predecessor (directly, or through the
predecessor's echoes: D -> echo -> D/A). live_values() then propagates the
superseded doc's death onto the successor, every value dies, and
corroborate()'s resurrect-all fallback quietly adopts the stale figure by
dict-order tie-break.

Two guards under test (both in corroboration.py):
  1. build_attribute_graph(): no derivation edges between two docs in the
     SAME authority chain (versions are related by supersession, not copying);
  2. live_values(): propagated death marks open-web ECHOES only -- a doc
     that belongs to an authority chain is never an echo of anything.
"""
from datetime import datetime

from webagg.corroboration import QTable, corroborate
from webagg.type_defs import Mention, Source

# ~28 tokens of shared form boilerplate: enough to trip the 25-token
# verbatim-run copy signal between any two docs that contain it, which is
# exactly what real Form D / D/A pairs look like (issuer block, addresses).
BOILER = ("issuer name acme corporation street address one rocket road city "
          "hawthorne state california zip 90250 phone 310 363 6000 industry "
          "group aerospace exemption 506b minimum investment accepted zero")

T_D = datetime(2021, 2, 23)
T_ECHO = datetime(2021, 3, 10)      # after the D, BEFORE the D/A: creates
T_DA = datetime(2021, 4, 14)        # the echo -> D/A copy-edge route


def _src(sid, t, text, *, chain=None, doc_type=None, anchored=False,
         sclass="blog", domain="example.com"):
    return Source(
        source_id=sid, url=f"https://{domain}/{sid}", domain=domain,
        fetch_time=t, publish_time=t, title=sid, main_text=text,
        formulation_id="test", source_class=sclass,
        authority_chain_id=chain, doc_type=doc_type,
        identity_anchored=anchored)


def _mention(src, value):
    return Mention(
        mention_id=Mention.make_id(src.source_id, "Acme", "funding_round",
                                   "amount", value, "A", src.source_id),
        source_id=src.source_id, entity_surface="Acme",
        record_kind="funding_round", attribute="amount", value=value,
        passage=f"amount {value}", extracted_at=src.publish_time,
        t_asof=src.publish_time, value_num=float(value), currency="USD",
        extractor_id="A", self_conf=1.0, accepted=True)


def test_amendment_survives_its_own_death_sentence():
    """The successor must never inherit the death it issued (Lemma 1:
    a superseded value is DISQUALIFIED; the superseding value must live)."""
    d = _src("d", T_D, BOILER + " total amount sold 850000000",
             chain="edgar:1:D", doc_type="Form D", anchored=True,
             sclass="regulatory", domain="sec.gov")
    da = _src("da", T_DA, BOILER + " amendment total amount sold 1164000000",
              chain="edgar:1:D", doc_type="Form D/A", anchored=True,
              sclass="regulatory", domain="sec.gov")
    # a stale echo: near-copy of the D (shares the boilerplate), timestamped
    # between D and D/A so the copy detector also finds echo -> D/A
    echo = _src("echo", T_ECHO, BOILER + " total amount sold 850000000 wow",
                domain="blog.example.com")

    lookup = {s.source_id: s for s in (d, da, echo)}
    mbv = {"850000000": [_mention(d, "850000000"), _mention(echo, "850000000")],
           "1164000000": [_mention(da, "1164000000")]}
    cv = corroborate(mbv, lookup, QTable())

    # the amended value is adopted; the old value and its echo are excluded
    assert cv.value == "1164000000", (
        "stale value adopted: the amendment was killed as a derivation "
        "descendant of the doc it supersedes (resurrect-all fallback)")
    assert cv.n_dead_excluded == 2          # the superseded D + its echo
    assert "850000000" not in cv.competing  # disqualified, not outvoted
