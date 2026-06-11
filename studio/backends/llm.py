"""Provider-agnostic proposer backend via LiteLLM.

Subclasses :class:`GeminiBackend` to reuse its agentic Tier-A loop and Tier-B
JSON contract verbatim — only the transport changes. ``litellm.completion`` is a
*stateless* call that speaks any provider ("gpt-5.4", "gemini/gemini-3.5-flash",
"anthropic/claude-...", "ollama/...") behind one OpenAI-shaped response, so the
inherited ``run_agent``/``prompt_json``/``_assistant_dict`` work unchanged. We
keep the same retry + thinking-model guard as the parent, and accumulate real
USD cost from ``litellm.completion_cost`` instead of the parent's flat price.

``drop_params=True`` lets LiteLLM paper over per-provider param quirks
(``max_tokens`` vs ``max_completion_tokens``, unsupported ``temperature``).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .gemini import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TURNS,
    GeminiBackend,
    GeminiBackendError,
)


class LLMBackend(GeminiBackend):
    name = "litellm"

    def __init__(
        self,
        *,
        tier_a_model: str,
        tier_b_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_retries: int = 5,
        log_dir: Path | None = None,
    ) -> None:
        # NB: do not call super().__init__ — it would build an OpenAI client and
        # require an api key. litellm.completion is stateless, so we set up the
        # same state the inherited methods read, by hand.
        self.tier_a_model = tier_a_model
        self.tier_b_model = tier_b_model or tier_a_model
        self.base_url = base_url
        self.api_key = api_key
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0

    # --- usage / cost accounting (thread-safe) ---

    def _track(self, usage) -> None:
        if not usage:
            return
        with self._lock:
            self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def _track_cost(self, resp) -> None:
        import litellm

        try:
            cost = litellm.completion_cost(completion_response=resp) or 0.0
        except Exception:  # noqa: BLE001 — accounting must never crash a call
            cost = 0.0
        with self._lock:
            self._cost_usd += float(cost)

    def token_usage(self) -> dict:
        p, c = self._toks()
        with self._lock:
            cost = self._cost_usd
        return {"prompt_tokens": p, "completion_tokens": c, "cost_usd": round(cost, 6)}

    # --- low-level completion via litellm (same retry + thinking guard) ---

    def _complete(self, messages, *, model, tools=None, tag="", max_tokens=DEFAULT_MAX_TOKENS):
        import litellm

        last = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    max_tokens=max_tokens,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    drop_params=True,
                )
            except Exception as e:  # noqa: BLE001 — classify then re-raise
                last = e
                if self._is_retryable(e) and attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise GeminiBackendError(f"litellm error (tag={tag}): {e}") from e
            self._track(getattr(resp, "usage", None))
            self._track_cost(resp)
            choice = resp.choices[0]
            msg = choice.message
            # Thinking-model guard: a too-small token budget can be consumed
            # entirely by reasoning, yielding empty output (finish_reason=length).
            if choice.finish_reason == "length" and not (msg.content or msg.tool_calls):
                if max_tokens < 65_536:
                    max_tokens *= 2
                    last = GeminiBackendError("empty output (finish_reason=length); raising max_tokens")
                    continue
            self._log(tag, resp)
            return resp
        raise GeminiBackendError(f"litellm completion failed (tag={tag}): {last}")
