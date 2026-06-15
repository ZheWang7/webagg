from webagg.search import SerperBackend
from webagg.fetch import fetch_url
from webagg.extract import is_relevant, extract_mentions

query = "Series A funding rounds for AI startups in 2024"
hits = SerperBackend().search(query, k=3)
print("search hits:", [h["url"] for h in hits])

src = fetch_url(hits[0]["url"], formulation_id="f0")
print("fetched:", src.title if src else "None (page unusable, try hits[1])")

if src:
    print("relevant?", is_relevant(src, query))
    for m in extract_mentions(src, query)[:3]:
        print(m.attribute, "=", m.value, "|", m.passage[:60])
