"""
Minimal wrapper around Groq's chat API. Groq's API is OpenAI-compatible, so
we use the `openai` client library but point it at Groq's servers -- that's
why you'll see `base_url` set below instead of just using OpenAI directly.

This is the ONLY file in the project that calls the LLM. Every other file
calls call_llm_text() / call_llm_json() instead of touching the client
directly -- that's what makes swapping providers later a one-file change.
"""
import os
import json
from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def call_llm_text(system: str, user: str, max_tokens: int = 500) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def call_llm_json(system: str, user: str, max_tokens: int = 500) -> dict:
    """Call the model and parse a JSON object out of its reply."""
    strict_system = (
        system
        + "\n\nRespond with ONLY a single valid JSON object. No markdown fences, "
        "no commentary before or after it."
    )
    raw = call_llm_text(strict_system, user, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw reply: {raw!r}") from e