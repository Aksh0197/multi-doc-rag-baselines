from __future__ import annotations

import os

# Global circuit breaker — tripped by any 401/402 so every component stops immediately
_LLM_DISABLED = False
_LLM_DISABLE_REASON = ""


def _openrouter_model(model: str) -> str:
    """Map Anthropic model names to OpenRouter model IDs."""
    mapping = {
        "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4.5",
        "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
        "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
        "claude-opus-4-7": "anthropic/claude-opus-4.7",
    }
    return mapping.get(model, "anthropic/claude-haiku-4.5")


def make_llm_call(prompt: str, model: str, max_tokens: int) -> str:
    """
    Unified LLM call supporting OPENROUTER_API_KEY or ANTHROPIC_API_KEY.
    Trips a global circuit breaker on 401/402 so the whole run stops retrying.
    """
    global _LLM_DISABLED, _LLM_DISABLE_REASON
    if _LLM_DISABLED:
        raise RuntimeError(f"LLM calls disabled: {_LLM_DISABLE_REASON}")

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    try:
        if groq_key and len(groq_key) > 20:
            return _call_groq(prompt, max_tokens, groq_key)
        elif anthropic_key and len(anthropic_key) > 20:
            return _call_anthropic(prompt, model, max_tokens, anthropic_key)
        elif openrouter_key and len(openrouter_key) > 20:
            return _call_openrouter(prompt, model, max_tokens, openrouter_key)
        else:
            raise EnvironmentError(
                "No valid API key found. Set GROQ_API_KEY (free at console.groq.com), "
                "OPENROUTER_API_KEY, or ANTHROPIC_API_KEY."
            )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "402" in msg or "authentication" in msg.lower() or "insufficient credits" in msg.lower():
            _LLM_DISABLED = True
            _LLM_DISABLE_REASON = msg[:120]
            print(f"\n[LLM] Circuit breaker tripped — disabling all LLM calls for this run.\n  Reason: {_LLM_DISABLE_REASON}\n")
        raise


def _call_openrouter(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    import urllib.request
    import urllib.error
    import json as _json

    or_model = _openrouter_model(model)
    payload = _json.dumps({
        "model": or_model,
        "max_completion_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rag-paper-baselines",
            "X-Title": "RAG Paper Baselines",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
    return data["choices"][0]["message"]["content"].strip()


def _call_groq(prompt: str, max_tokens: int, api_key: str) -> str:
    """Free tier via Groq — uses llama-3.1-8b-instant with retry on 429."""
    import time
    import urllib.request
    import urllib.error
    import json as _json

    payload = _json.dumps({
        "model": "llama-3.1-8b-instant",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    for attempt in range(4):  # up to 3 retries on 429
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 429 and attempt < 3:
                wait = 4 * (attempt + 1)  # 4s, 8s, 12s
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq HTTP {e.code}: {body}") from e
    raise RuntimeError("Groq: max retries exceeded on 429")


def _call_anthropic(prompt: str, model: str, max_tokens: int, api_key: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()
