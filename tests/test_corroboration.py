"""Section 7.4 sanity tests for the corroboration layer."""
import hashlib
from datetime import datetime

from webagg.type_defs import Source, Mention
from webagg.corroboration import (
    shingle_jaccard, longest_common_verbatim_tokens,
    derivation_edge, corroborate,
)


def make_source(domain: str, date: str, text: str) -> Source:
    """A minimal Source. 'date' is ISO 'YYYY-MM-DD'; used for publish_time."""
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


# --- helper-level checks ----------------------------------------------------
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
    late = make_source("b.com", "2025-01-05", "Acme closed a 40M Series B today")
    # late derives from early (forward) -> True; early-from-late (backward) -> False
    assert derivation_edge(early, late, early.main_text, late.main_text) is True
    assert derivation_edge(late, early, late.main_text, early.main_text) is False


# --- the key invariant: echoes don't inflate belief -------------------------
def test_echo_does_not_change_belief():
    sec = make_source("sec.gov", "2025-01-08", "Acme raised 40M Series B in a filing.")
    pr = make_source("prnewswire.com", "2025-01-03",
                     "Acme Corp closed a 40M Series B today, led by Foo Ventures.")
    copies = [make_source(f"blog{i}.example", "2025-01-04",
                          "Acme Corp closed a 40M Series B today, led by Foo Ventures.")
              for i in range(10)]

    base = corroborate({"$40M": [m_of(sec)]}, _lookup(sec))
    flooded = corroborate(
        {"$40M": [m_of(sec), m_of(pr)] + [m_of(c) for c in copies]},
        _lookup(sec, pr, *copies),
    )

    # SEC alone vs SEC + (PR with 10 echoes that collapse to ONE component):
    assert flooded.nu == 2                       # two independent witnesses
    assert flooded.belief > base.belief          # a bit more corroboration
    assert flooded.belief < 0.999                # but NOT 12x worth
    print(f"\nbase.belief={base.belief:.4f}  "
          f"flooded.belief={flooded.belief:.4f}  flooded.nu={flooded.nu}  "
          f"comp_sizes={flooded.component_sizes}")


def test_competing_values_pick_strongest_witness():
    sec = make_source("sec.gov", "2025-01-08", "Acme raised 40M Series B.")
    blog = make_source("blog.example", "2025-01-02", "Acme raised 50M Series B.")
    out = corroborate(
        {"$40M": [m_of(sec, "$40M")], "$50M": [m_of(blog, "$50M")]},
        _lookup(sec, blog),
    )
    assert out.value == "$40M"          # SEC (0.97) beats unknown blog (0.50)
    assert "$50M" in out.competing
