"""
model_client.py — one interface for every model in the study.

Two providers behind one roof:
  - Anthropic  (model starts with "claude")  -> native Messages API, supports prompt caching
  - OpenRouter (everything else)             -> OpenAI-compatible API, open-weights models

Single-shot text completion for now (monitor + faithfulness judge use this).
The agent tool-use loop (agent.py) builds on these same clients.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import anthropic
from openai import OpenAI

# --- clients ---------------------------------------------------------------
# timeouts + retries so one slow/hung request can't stall a whole run
_anthropic = anthropic.Anthropic(timeout=90, max_retries=2)  # reads ANTHROPIC_API_KEY from env

_or_key = os.environ.get("OPENROUTER_API_KEY", "")
_openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_or_key,
                     timeout=90, max_retries=2)


def _is_claude(model: str) -> bool:
    return model.lower().startswith("claude")


def complete(model, system, user, cache=False, max_tokens=1024):
    """Send one system+user prompt, get back plain text. Routes by model name."""
    if _is_claude(model):
        # Prompt caching: mark the system prefix as cacheable (Anthropic only).
        sys_blocks = [{"type": "text", "text": system}]
        if cache:
            sys_blocks[-1]["cache_control"] = {"type": "ephemeral"}
        resp = _anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=sys_blocks,
            messages=[{"role": "user", "content": user}],
        )
        # collect text blocks (a thinking block can lead; don't assume content[0])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    else:
        resp = _openrouter.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


# --- smoke test / connectivity check --------------------------------------
if __name__ == "__main__":
    print("== Anthropic ==")
    try:
        print("reply:", complete("claude-haiku-5-5", "Reply with exactly: OK", "ping"))
    except Exception as e:
        print("Anthropic FAILED:", type(e).__name__, e)

    print("\n== OpenRouter ==")
    if not _or_key:
        print("no OPENROUTER_API_KEY in .env yet — skipping")
    else:
        try:
            models = _openrouter.models.list()
            print(f"key OK — {len(models.data)} models available")
            print("sample ids:")
            for m in list(models.data)[:15]:
                print("  ", m.id)
        except Exception as e:
            print("OpenRouter FAILED:", type(e).__name__, e)
