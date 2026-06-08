import os, json, time
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from . import config  # importing config runs load_dotenv(), so the key below is populated

_client_anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # cheap; bump for harder prompts
DEFAULT_MODEL_OPENAI = "gpt-5-nano"

# Anthropic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm2(*, system: str, user: str, model: str = DEFAULT_MODEL,
             max_tokens: int = 1024, json_mode: bool = True) -> dict:
    t0 = time.time()
    resp = _client_anthropic.messages.create(
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


# OpenAI
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(*, system: str, user: str, model: str = DEFAULT_MODEL_OPENAI,
             max_tokens: int = 4096, json_mode: bool = True) -> dict:
    t0 = time.time()
    kwargs = dict(
        model=model,
        max_completion_tokens=max_tokens,   # GPT-5 uses this, NOT max_tokens
        reasoning_effort="low",             # extraction tasks don't need deep reasoning
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}  # force valid JSON
    resp = _client_openai.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        # a reasoning model can burn the whole budget on hidden reasoning tokens
        raise ValueError(
            f"Empty response (finish_reason={choice.finish_reason}); "
            f"raise max_tokens so there's room for output after reasoning."
        )

    if json_mode:
        if text.startswith("```"):
            text = text.strip("`")
        text = text[text.find("{"):]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            if choice.finish_reason == "length":
                raise ValueError(
                    f"LLM output was truncated at max_tokens={max_tokens}; "
                    f"raise max_tokens or reduce the input size."
                ) from e
            raise ValueError(f"LLM did not return JSON: {text[:300]}") from e
    else:
        payload = {"text": text}

    return {
        "payload": payload,
        "model": model,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "latency_s": time.time() - t0,
    }
