import httpx, trafilatura
from datetime import datetime
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential
from .type_defs import Source

UA = "webagg-research/0.1 (mailto:you@example.com)"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def fetch_url(url: str, formulation_id: str) -> Source | None:
    r = httpx.get(url, headers={"User-Agent": UA},
                  follow_redirects=True, timeout=20.0)
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
        return None
    main_text = trafilatura.extract(r.text) or ""
    if len(main_text) < 200:           # too short to be useful
        return None
    meta = trafilatura.extract_metadata(r.text)
    fetch_time = datetime.utcnow()
    return Source(
        source_id=Source.make_id(url, fetch_time),
        url=url, domain=urlparse(url).netloc,
        fetch_time=fetch_time,
        publish_time=meta.date if meta and meta.date else None,
        title=meta.title if meta else None,
        main_text=main_text[:20000],   # cap for LLM context
        formulation_id=formulation_id,
    )
