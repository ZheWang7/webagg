"""Offline smoke test for the wired pipeline (impl guide Sec. 10.7 / 11.1).

Populates a fixture run-DB by hand, then runs resolve_and_aggregate with a
stub cluster_fn -- zero network, zero LLM calls, zero API keys touched.
"""
from datetime import datetime

from webagg.canonicalize import canonicalize_value
from webagg.pipeline import resolve_and_aggregate
from webagg.storage import get_session
from webagg.type_defs import Mention, Source

QUERY_ATTRS = {"amount", "date", "employees"}


def make_source(sid, domain, day, text):
    t = datetime(2025, 1, day, 12, 0, 0)
    return Source(source_id=sid, url=f"https://{domain}/x", domain=domain,
                  fetch_time=t, publish_time=t, title=None,
                  main_text=text, formulation_id="f1")


def make_mention(src, attr, value, passage, surface="Acme Corp"):
    return Mention(mention_id=f"{src.source_id}:{attr}:{hash(value) & 0xffff}",
                   source_id=src.source_id, entity_surface=surface,
                   record_kind="funding_round", attribute=attr, value=value,
                   passage=passage, extracted_at=src.fetch_time)


PR_PASSAGE = ("Acme Corp today announced it has closed a forty million dollar "
              "Series B financing round led by a16z")


def build_fixture_db(tmp_path):
    session = get_session(str(tmp_path / "fixture.sqlite"))

    # independent origin #1: the SEC filing (q = 0.97), distinct wording
    sec = make_source("sec1", "sec.gov", 8,
                      "Form D filing. Total amount sold: $40,000,000.")
    # independent origin #2: PR release (Jan 3) ...
    pr = make_source("pr1", "prnewswire.com", 3, PR_PASSAGE)
    # ... plus a blog copying the PR verbatim a day later -> derivation edge
    # (near-duplicate + later timestamp), SAME component as pr (Def. 8)
    echo = make_source("blog1", "blog1.example", 4, PR_PASSAGE)
    # a lone low-quality blog asserting a WRONG amount
    wrong = make_source("blog2", "blog2.example", 5,
                        "Sources say Acme actually raised $42M in the round.")
    # LinkedIn page holding employees -- and NEVER naming Acme (guard case)
    li = make_source("li1", "linkedin.com", 6,
                     "This fast-growing logistics startup now has 87 employees.")

    sources = [sec, pr, echo, wrong, li]
    mentions = [
        # amount: three spellings of the same value + one competitor
        make_mention(sec, "amount", "$40M", sec.main_text),
        make_mention(pr, "amount", "40 million USD", PR_PASSAGE),
        make_mention(echo, "amount", "$0.04B", PR_PASSAGE),
        make_mention(wrong, "amount", "$42M", wrong.main_text),
        # date: contributed by two classes with distinct wording
        make_mention(sec, "date", "2025-01-02", "Date of first sale: 2025-01-02"),
        make_mention(pr, "date", "2025-01-02", "the round closed on January 2"),
        # employees: SOCIAL only -> the fragmenting attribute
        make_mention(li, "employees", "87", "now has 87 employees"),
    ]
    for obj in sources + mentions:
        session.add(obj.to_row())
    session.commit()
    return session


def stub_cluster(mentions, sources):
    """ER stand-in: single-entity fixture, so everything is ent_00000.
    (Same injection pattern as the ER adjudicator / schema relevance_fn.)"""
    return {m.mention_id: "ent_00000" for m in mentions}


def test_canonicalize_value_spellings():
    assert canonicalize_value("$40M") == "40000000"
    assert canonicalize_value("40 million USD") == "40000000"
    assert canonicalize_value("$0.04B") == "40000000"
    assert canonicalize_value("40,000,000") == "40000000"
    assert canonicalize_value("44") == "44"
    assert canonicalize_value("2025-01-02") == "2025-01-02"   # dates untouched
    assert canonicalize_value("Indiana  Pacers") == "indiana pacers"
    assert canonicalize_value("Series B") == "series b"       # not a number


def test_resolve_and_aggregate_offline(tmp_path):
    session = build_fixture_db(tmp_path)
    res = resolve_and_aggregate(session, run_id="offline_test",
                                query_attributes=QUERY_ATTRS,
                                aggregate_attr="amount",
                                eps=0.10, eps_er=0.05,
                                cluster_fn=stub_cluster)

    # exactly one resolved record: (ent_00000, funding_round)
    assert len(res["records"]) == 1
    rec = res["records"][0]

    # fragmentation: employees held only by SOCIAL -> complementary (Def. 16)
    assert rec["frag_case"] == "fragmented"
    (_, rep, _), = res["reports"]
    assert rep.fragmenting_attrs == {"employees"}

    # canonicalization + corroboration on amount: 3 spellings merged into one
    # candidate; PR + echo collapse to ONE witness, SEC is the second (nu=2);
    # the widely-spelled $40M beats the lone $42M (Def. 10)
    amount = rec["attributes"]["amount"]
    assert amount.value == "40000000"
    assert amount.nu == 2
    assert abs(amount.belief - (1 - (1 - 0.97) * (1 - 0.50))) < 1e-9   # 0.985
    assert "42000000" in amount.competing

    # 6.9 guard: LinkedIn never names Acme -> q(linkedin)=0.5 halved to 0.25
    assert abs(rec["attributes"]["employees"].belief - 0.25) < 1e-9

    # aggregation (Corollary 2): SUM = 40M; recall term = eps * SUM
    assert res["answer"] == 40_000_000.0
    ci = res["ci"]
    assert abs(ci["recall_term"] - 0.10 * 40_000_000.0) < 1e-6
    assert ci["ci_total"] >= ci["recall_term"] > 0
