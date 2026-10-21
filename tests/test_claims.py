from webaggs.claims import ClaimsEngine
from webaggs.types import Claim


def _claim(domain, kind, value, src="s1"):
    return Claim(claim_id=f"{src}:{domain}:{kind}:{value}", source_id=src,
                 domain=domain, stratum="p", kind=kind, value=value)


def test_claim_needs_two_domains():
    """Wrong claims are contained, not trusted (webaggr.pdf §5.2): one domain
    — even repeated — never corroborates; echo on a domain is one vote."""
    ce = ClaimsEngine()
    ce.ingest(_claim("a.test", "COUNT", 3))
    ce.ingest(_claim("a.test", "COUNT", 3, src="s2"))   # echo: same domain
    assert ce.corroborated("p", "COUNT") is None
    ce.ingest(_claim("b.test", "COUNT", 3))
    assert ce.corroborated("p", "COUNT") == 3


def test_count_closure_and_gap_direction():
    ce = ClaimsEngine()
    ce.ingest(_claim("a.test", "COUNT", 3))
    ce.ingest(_claim("b.test", "COUNT", 3))
    st = ce.checksum("p", n_records=2, assembled_sum=1000)
    assert not st.certified and st.gap == 1              # one season missing
    fms = ce.gap_formulations("p", "Testo Sampleson", st.gap, {"2019-20"})
    assert fms and all(f.gap_directed for f in fms)      # targeted searches
    st = ce.checksum("p", n_records=3, assembled_sum=1000)
    assert st.certified and st.kind == "COUNT"           # exact closure (Thm 4)


def test_sum_checksum_and_delta_plus():
    ce = ClaimsEngine()
    ce.ingest(_claim("a.test", "SUM", 1450))
    ce.ingest(_claim("b.test", "SUM", 1450))
    st = ce.checksum("p", n_records=2, assembled_sum=1062)
    assert st.delta_plus == 388                          # missing-value bound
    assert not st.certified                              # outside tol_rel
    st = ce.checksum("p", n_records=3, assembled_sum=1450)
    assert st.certified and st.kind == "SUM"


def test_conflicting_claim_values_do_not_corroborate():
    ce = ClaimsEngine()
    ce.ingest(_claim("a.test", "COUNT", 3))
    ce.ingest(_claim("b.test", "COUNT", 5))              # disagreement
    assert ce.corroborated("p", "COUNT") is None
