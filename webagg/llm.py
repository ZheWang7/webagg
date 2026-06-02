import os, json, time
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from . import config  # importing config runs load_dotenv(), so the key below is populated

_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheap; bump for harder prompts


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(*, system: str, user: str, model: str = DEFAULT_MODEL,
             max_tokens: int = 1024, json_mode: bool = True) -> dict:
    t0 = time.time()
    resp = _client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text.strip()
    if json_mode:
        # strip code fences if the model wrapped its JSON in them
        if text.startswith("```"):
            text = text.strip("`")
        text = text[text.find("{"):]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            if resp.stop_reason == "max_tokens":
                raise ValueError(
                    f"LLM output was truncated at max_tokens={max_tokens}; "
                    f"raise max_tokens or reduce the input size."
        ) from e
            raise ValueError(f"LLM did not return JSON: {text[:200]}") from e
    else:
        payload = {"text": text}
    return {
        "payload": payload,
        "model": model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "latency_s": time.time() - t0,
    }
