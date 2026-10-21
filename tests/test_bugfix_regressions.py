"""One regression test per bug found in the verification pass (2026-10-20)."""
from webaggs.canon import canonical_points
from webaggs.claims import ClaimsEngine
from webaggs.types import Claim


def _claim(domain, kind, value):
    return Claim(claim_id=f"{domain}:{kind}:{value}", source_id="s",
                 domain=domain, stratum="p", kind=kind, value=value)


def test_ppg_decimals_are_rejected_not_truncated():
    """Bug 1: 28.4 is a per-game figure; truncating it to 28 would assert a
    bogus season total. The typed validator must reject every fractional."""
    assert canonical_points(28.4) is None
    assert canonical_points("3.5 ppg") is None
    assert canonical_points("he averaged 28.4 points") is None
    assert canonical_points(450.0) == 450          # integral float still fine
    assert canonical_points("1,138") == 1138
    assert canonical_points(True) is None          # bool is not a point total


def test_conflicting_corroborated_claims_refuse():
    """Bug 2: two values tied at the top domain support = a contradiction;
    refusing beats an arbitrary dict-order winner."""
    ce = ClaimsEngine()
    for d in ("a.test", "b.test"):
        ce.ingest(_claim(d, "COUNT", 3))
    for d in ("c.test", "d.test"):
        ce.ingest(_claim(d, "COUNT", 5))
    assert ce.corroborated("p", "COUNT") is None   # 2-vs-2: refuse
    ce.ingest(_claim("e.test", "COUNT", 3))        # 3-vs-2: strict winner
    assert ce.corroborated("p", "COUNT") == 3


def test_sum_cannot_certify_against_a_contradicting_count():
    """Bug 3: a SUM within tolerance must not certify while a corroborated
    COUNT disagrees with the record count (e.g. spurious records inflating
    toward a lifetime total). Checksum vs checksum contradiction = flag."""
    ce = ClaimsEngine()
    for d in ("a.test", "b.test"):
        ce.ingest(_claim(d, "COUNT", 3))
        ce.ingest(_claim(d, "SUM", 1000))
    # 4 records summing to the claimed 1000, but COUNT says 3: no certificate
    st = ce.checksum("p", n_records=4, assembled_sum=1000)
    assert not st.certified
    # with the count satisfied, the SUM tolerance checksum works as before
    st = ce.checksum("p", n_records=3, assembled_sum=1000)
    assert st.certified and st.kind == "COUNT"     # closure wins first
