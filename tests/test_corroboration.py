"""Section 8.4 sanity tests for the corroboration layer (SIGMOD guide).

Four behavioural guarantees, straight from the guide:
  1. echo invariance          -- copies don't add witnesses;
  2. stale-echo disqualified  -- supersession kills a version and its echoes,
                                 it never gets outvoted by them;
  3. margin matches paper     -- the "two-line test that catches a sign
                                 error every time";
  4. unanchored cap           -- a copy farm agreeing with itself never
                                 exceeds a qbar-belief contribution.
Plus the helper-level checks kept from the previous version of this file.
"""
import hashlib
from datetime import datetime

from webagg.type_defs import Source, Mention
from webagg.corroboration import (
    shingle_jaccard, longest_common_verbatim_tokens, derivation_edge,
    supersession_edges, is_amendment, corroborate, margin, QTable,
)


# --- fixtures ---------------------------------------------------------------
def make_source(domain: str, date: str, text: str, *,
                source_class: str | None = None,
                anchored: bool = False,
                chain: str | None = None,
                doc_type: str | None = None) -> Source:
    """A minimal Source. 'date' is ISO 'YYYY-MM-DD[THH:MM]'."""
    ts = datetime.fromisoformat(date)
    sid = hashlib.sha256(f"{domain}|{date}|{text}".encode()).hexdigest()[:16]
    return Source(
        source_id=sid,
        url=f"https://{domain}/article",
        domain=domain,
        fetch_time=ts,
        publish_time=ts,
        title=None,
        main_text=text,
        formulation_id="test",
        source_class=source_class,
        identity_anchored=anchored,
        authority_chain_id=chain,
        doc_type=doc_type,
    )


def m_of(source: Source, value: str = "$40M") -> Mention:
    """A Mention asserting 'value', whose passage IS the source's text."""
    return Mention(
        mention_id=f"{source.source_id}:amount:{value}",
        source_id=source.source_id,
        entity_surface="Acme",
        record_kind="funding_round",
        attribute="amount",
        value=value,
        passage=source.main_text,
        extracted_at=datetime.utcnow(),
    )


def _lookup(*sources: Source) -> dict[str, Source]:
    return {s.source_id: s for s in sources}


# --- helper-level checks (kept from the 7.4 file) ---------------------------
def test_shingle_jaccard_identical_text_is_one():
    t = "Acme Corp closed a 40M Series B today in San Francisco"
    assert shingle_jaccard(t, t) == 1.0
    assert shingle_jaccard(t, "totally unrelated wording here entirely") < 0.85


def test_longest_common_verbatim_run():
    a = "the quick brown fox jumps over the lazy dog"
    b = "yesterday the quick brown fox sat down"
    assert longest_common_verbatim_tokens(a, b) == 4  # "the quick brown fox"


def test_temporal_direction_blocks_backward_edge():
    early = make_source("a.com", "2025-01-01", "Acme closed a 40M Series B today")
    late = make_source("b.com", "2025-01-02", "Acme closed a 40M Series B today")
    # b.com (later) can have copied a.com; a.com cannot have copied b.com
    assert derivation_edge(early, late)
    assert not derivation_edge(late, early)


def test_is_amendment():
    assert is_amendment("Form D/A") and is_amendment("10-K/A")
    assert is_amendment("updated") and not is_amendment("Form D")


# --- 8.4 test 1: echo invariance --------------------------------------------
def _echo_setup(n_blogs: int):
    """One SEC filing vs. the same value from a PR + n near-duplicate blogs.

    The SEC text and the PR/blog text are worded differently (no derivation
    edge across them); the blogs copy the PR verbatim with later timestamps.
    Expect exactly TWO independent origins regardless of n_blogs.
    """
    sec = make_source(
        "sec.gov", "2025-01-01",
        "Form D filed. Total offering amount sold to date: forty million "
        "dollars for the issuer Acme Incorporated of Delaware.",
        source_class="registry", anchored=True)
    pr_text = ("Acme announces the successful close of its forty million "
               "Series B financing round led by Example Capital, with "
               "participation from prior investors, the company said today.")
    pr = make_source("prnewswire.com", "2025-01-02", pr_text,
                     source_class="news")
    blogs = [
        make_source(f"blog{i}.com", f"2025-01-{3 + i:02d}", pr_text,
                    source_class="blog")
        for i in range(n_blogs)
    ]
    all_srcs = [sec, pr, *blogs]
    mbv = {"$40M": [m_of(s) for s in all_srcs]}
    return mbv, _lookup(*all_srcs)


def test_echo_invariance():
    """nu = 2 (SEC + press cluster) and belief invariant in #copies."""
    cv10 = corroborate(*_echo_setup(10), QTable())
    cv3 = corroborate(*_echo_setup(3), QTable())
    assert cv10.nu == 2 and cv3.nu == 2
    assert abs(cv10.belief - cv3.belief) < 1e-12    # copies add NOTHING


# --- 8.4 test 2: stale echo disqualified, not outvoted ----------------------
def test_stale_echo_disqualified_not_outvoted():
    """Form D $35M + 13 echoes vs. lone Form D/A $40M (same chain).

    14 mentions say $35M, one says $40M. Majority-vote adopts $35M; the
    corroboration layer must adopt $40M because the D/A structurally
    supersedes the D, and every echo of the D dies with it.
    """
    d_text = ("Form D filed with the Securities and Exchange Commission. "
              "Total amount sold: thirty five million dollars, Acme Inc.")
    d = make_source("sec.gov", "2025-01-01", d_text,
                    source_class="registry", anchored=True,
                    chain="edgar:0001234567:D", doc_type="Form D")
    pr = make_source("prnewswire.com", "2025-01-02", d_text,   # verbatim echo
                     source_class="news")
    blogs = [make_source(f"blog{i}.com", f"2025-01-{3 + i:02d}", d_text,
                         source_class="blog") for i in range(12)]
    da = make_source(
        "sec.gov", "2025-02-01T12:00",
        "Form D/A amendment filed. Total amount sold updated: forty "
        "million dollars, Acme Inc.",
        source_class="registry", anchored=True,
        chain="edgar:0001234567:D", doc_type="Form D/A")

    src = _lookup(d, pr, *blogs, da)
    mbv = {"$35M": [m_of(s, "$35M") for s in (d, pr, *blogs)],
           "$40M": [m_of(da, "$40M")]}
    cv = corroborate(mbv, src, QTable())
    assert cv.value == "$40M"
    assert cv.n_dead_excluded == 14        # D + PR + 12 blogs, all auditable
    assert cv.version_id == 1              # second doc of the chain
    assert "$35M" not in cv.competing      # disqualified, not merely losing


def test_supersession_edge_detected():
    d = make_source("sec.gov", "2025-01-01", "Form D text",
                    anchored=True, chain="c1", doc_type="Form D",
                    source_class="registry")
    da = make_source("sec.gov", "2025-02-01", "Form D/A amendment text",
                     anchored=True, chain="c1", doc_type="Form D/A",
                     source_class="registry")
    edges = supersession_edges([d, da])
    assert (d.source_id, da.source_id, "form_amendment") in edges


# --- 8.4 test 3: margin matches paper ---------------------------------------
def test_margin_matches_paper():
    """The guide's 'two-line test that catches a sign error every time'."""
    assert margin(0.9964, 0.51, 0.30) == 13
    assert margin(0.9964, 0.0, 0.30) == 15


# --- 8.4 test 4: unanchored self-consistency capped -------------------------
def test_unanchored_self_consistency_capped():
    """A 20-page copy farm agreeing with itself must not exceed a qbar
    belief contribution: the pages collapse into ONE component, and an
    unanchored component is capped at qbar = 0.30."""
    farm_text = ("Breaking exclusive: sources confirm Acme has quietly "
                 "raised forty million in fresh capital from undisclosed "
                 "backers, according to people familiar with the matter.")
    farm = [make_source(f"copyfarm{i}.biz", f"2025-01-01T08:{i:02d}",
                        farm_text, source_class="blog")
            for i in range(20)]
    cv = corroborate({"$40M": [m_of(s) for s in farm]}, _lookup(*farm),
                     QTable())
    assert cv.nu == 1                       # 20 pages, one origin
    assert cv.belief <= 0.30 + 1e-9         # <= qbar, never more
