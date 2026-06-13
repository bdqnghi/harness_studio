"""Construct a Backend from a model string.

Every model goes through the provider-agnostic :class:`LLMBackend` (LiteLLM),
which accepts any litellm model string ("gpt-5.4", "gemini/gemini-3.5-flash",
"anthropic/claude-...", "ollama/...").

The import is deferred into the function so importing this module never pulls in
``litellm`` unless a backend is actually constructed.
"""

from __future__ import annotations

from .base import Backend


def make_backend(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    tier_a_model: str | None = None,
    tier_b_model: str | None = None,
    log_dir=None,
) -> Backend:
    from .llm import LLMBackend

    return LLMBackend(
        tier_a_model=tier_a_model or model,
        tier_b_model=tier_b_model or model,
        base_url=base_url,
        api_key=api_key,
        log_dir=log_dir,
    )
