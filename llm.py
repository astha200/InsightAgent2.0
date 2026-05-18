import os
import re
import json
from typing import Any

from langsmith import traceable

MODEL_OPUS = "claude-opus-4-7"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5-20251001"

_client = None


def _get_client():
    global _client
    if _client is None:
        from anthropic import Anthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. export it before running."
            )
        _client = Anthropic()
    return _client


@traceable(name="claude_call")
def call(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    cache_system: bool = True,
) -> str:
    client = _get_client()
    sys_param: Any = (
        [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if cache_system
        else system
    )
    resp = client.messages.create(
        model=model,
        system=sys_param,
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in: {text[:300]!r}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"Unbalanced JSON in: {text[start : start + 300]!r}")


def call_json(**kwargs) -> dict:
    return extract_json(call(**kwargs))
