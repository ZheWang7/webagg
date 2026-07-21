"""LLM wrapper (impl guide ch. 5): the ONLY module that imports provider SDKs.

SIGMOD-version requirements implemented here:
  1. strict-JSON output with ONE corrective re-prompt on parse failure
     (the model is shown its own bad output + the parse error);
  2. retry with exponential backoff (transport/API errors, via tenacity);
  3. every call logged to the measurements table (token counts, latency,
     model, purpose) -- cost auditing is an experiment, not an afterthought;
  4. model choice is a CONFIG key: cheap for relevance/adjudication,
     strong for extraction (config.MODEL_CHEAP / config.MODEL_STRONG).
"""
import os, json, time
from anthropic import Anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from . import config  # importing config runs load_dotenv(), so keys are populated

_client_anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_client_openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Measurement sink (guide ch. 5: "log ... to measurements").
# The pipeline registers a (session, run_id) pair at run start; until then,
# calls simply aren't logged, so unit tests and one-off scripts need no DB.
# ---------------------------------------------------------------------------
_SINK = {"session": None, "run_id": None, "step": 0}


def set_llm_logger(session, run_id: str):
    """Called once by the pipeline at run start."""
    _SINK["session"], _SINK["run_id"], _SINK["step"] = session, run_id, 0


def set_llm_step(step: int):
    """Called by the frontier loop so cost rows carry the agent step."""
    _SINK["step"] = step


def _log_call(purpose: str, model: str, in_tok: int, out_tok: int,
              latency: float, reprompted: bool):
    if _SINK["session"] is None:
        return
    from .metrics import log_measurement   # lazy: avoids circular import
    log_measurement(
        _SINK["session"], _SINK["run_id"], _SINK["step"], "llm_call",
        # value = total tokens (the cost driver); details go in extra
        float(in_tok + out_tok),
        extra={"purpose": purpose, "model": model,
               "input_tokens": in_tok, "output_tokens": out_tok,
               "latency_s": round(latency, 3), "reprompted": reprompted},
    )


def _parse_strict_json(text: str) -> dict:
    """Strip code fences / leading prose, then parse. Raises on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    text = text[text.find("{"):]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Main entry point (OpenAI backend -- single-API design; one provider at
# a time, per the project decision).
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(*, system: str, user: str, model: str | None = None,
             max_tokens: int = 4096, json_mode: bool = True,
             schema: dict | None = None, purpose: str = "") -> dict:
    """call_llm(system, user, schema=None) -> dict  (guide ch. 5 contract).

    model=None -> config.MODEL_CHEAP. Extraction call sites pass
    config.MODEL_STRONG explicitly; everything else defaults to cheap.
    schema: optional JSON schema; appended to the system prompt as an
    output contract (kept prompt-level for now: simple and portable).
    purpose: short tag ("relevance", "extraction", ...) for cost auditing.
    """
    model = model or config.MODEL_CHEAP
    if schema is not None:
        system = (system + "\n\nReturn ONLY a JSON object matching this "
                  "schema exactly:\n" + json.dumps(schema))

    def _once(messages, budget):
        kwargs = dict(
            model=model,
            max_completion_tokens=budget,       # GPT-5 uses this, NOT max_tokens
            reasoning_effort="low",             # extraction doesn't need deep reasoning
            messages=messages,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # force valid JSON
        return _client_openai.chat.completions.create(**kwargs)

    t0 = time.time()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    in_tok = out_tok = 0
    reprompted = False
    budget = max_tokens
    escalations = 0

    # BUDGET-ESCALATION LOOP. A reasoning model shares ONE completion cap
    # between hidden reasoning and visible output, so a dense page can fail
    # two ways with the same cause: (a) the whole cap goes to reasoning and
    # the text comes back EMPTY, or (b) the JSON is TRUNCATED mid-object.
    # Same disease, same cure: double the cap and call again -- paid only
    # when a page actually needs it. Tenacity (outside) still handles
    # transport errors; this loop handles budget errors; the corrective
    # re-prompt below handles malformed-but-complete JSON (guide ch. 5).
    while True:
        resp = _once(messages, budget)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        in_tok += resp.usage.prompt_tokens       # every attempt is real spend
        out_tok += resp.usage.completion_tokens
        truncated = (choice.finish_reason == "length")

        if text and not json_mode:
            payload = {"text": text}
            break
        if text and json_mode:
            try:
                payload = _parse_strict_json(text)
                break
            except json.JSONDecodeError as e:
                if not truncated:
                    # complete but malformed -> corrective re-prompt ONCE,
                    # showing the model its own output and the error
                    if reprompted:
                        raise ValueError(
                            f"LLM did not return JSON after re-prompt: "
                            f"{text[:300]}") from e
                    reprompted = True
                    messages += [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content":
                            f"That was not valid JSON ({e}). Reply again with "
                            f"ONLY the corrected JSON object -- no prose, no "
                            f"code fences."},
                    ]
                    continue
                # truncated JSON: fall through to budget escalation
        elif not text and not truncated:
            # empty for a reason money can't fix
            raise ValueError(
                f"Empty response (finish_reason={choice.finish_reason}).")

        # empty-or-truncated because of the cap -> escalate: base -> 2x -> 4x
        if escalations >= 2:
            raise ValueError(
                f"Empty/truncated response even at max_tokens={budget} "
                f"(finish_reason={choice.finish_reason}); the input may be "
                f"too large for one call.")
        escalations += 1
        budget *= 2

    latency = time.time() - t0
    _log_call(purpose, model, in_tok, out_tok, latency, reprompted)
    return {
        "payload": payload,
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "latency_s": latency,
    }


# ---------------------------------------------------------------------------
# Anthropic backend, kept as the alternate single provider. If you switch
# back to it, give it the same re-prompt + _log_call treatment as above.
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm2(*, system: str, user: str, model: str = "claude-haiku-4-5-20251001",
              max_tokens: int = 1024, json_mode: bool = True) -> dict:
    t0 = time.time()
    resp = _client_anthropic.messages.create(
        model=model, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = resp.content[0].text.strip()
    if json_mode:
        try:
            payload = _parse_strict_json(text)
        except json.JSONDecodeError as e:
            if resp.stop_reason == "max_tokens":
                raise ValueError(
                    f"LLM output was truncated at max_tokens={max_tokens}; "
                    f"raise max_tokens or reduce the input size.") from e
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
