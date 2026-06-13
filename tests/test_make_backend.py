"""Hermetic tests for the make_backend factory and LiteLLM backend.

No network: ``litellm.completion`` is patched with a fake that returns an
OpenAI-shaped response object. Asserts the factory builds an LLMBackend and that
the inherited Tier-B JSON contract works over the litellm transport.
"""

from __future__ import annotations

import json

import pytest

from studio import schemas
from studio.backends.factory import make_backend
from studio.backends.llm import LLMBackend


# --- minimal OpenAI-shaped fake of a litellm.completion response ---

class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Resp:
    def __init__(self, content, finish_reason="stop"):
        msg = _Msg(content=content)
        self.choices = [type("C", (), {"message": msg, "finish_reason": finish_reason})()]
        self.usage = _Usage()

    def model_dump_json(self, indent=None):
        return "{}"


def _fake_completion(content):
    def _call(**kwargs):
        _call.calls.append(kwargs)
        return _Resp(content)

    _call.calls = []
    return _call


@pytest.fixture
def patch_litellm(monkeypatch):
    """Patch studio.backends.llm.litellm.completion (and completion_cost)."""

    def _install(content='{"order": ["a", "b"]}'):
        import litellm

        fake = _fake_completion(content)
        monkeypatch.setattr(litellm, "completion", fake)
        monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0012)
        return fake

    return _install


# --- factory routing ---

def test_make_backend_returns_llm_for_openai_model():
    b = make_backend("gpt-5.4")
    assert isinstance(b, LLMBackend)
    assert b.tier_a_model == "gpt-5.4"
    assert b.tier_b_model == "gpt-5.4"


def test_make_backend_returns_llm_for_gemini_model():
    b = make_backend("gemini/gemini-3.5-flash")
    assert isinstance(b, LLMBackend)
    assert b.tier_a_model == "gemini/gemini-3.5-flash"


def test_make_backend_passes_through_overrides_and_urls():
    b = make_backend(
        "ollama/llama3",
        base_url="http://localhost:11434",
        api_key="k",
        tier_b_model="ollama/llama3-mini",
    )
    assert isinstance(b, LLMBackend)
    assert b.tier_a_model == "ollama/llama3"
    assert b.tier_b_model == "ollama/llama3-mini"
    assert b.base_url == "http://localhost:11434"
    assert b.api_key == "k"


# --- inherited Tier-B contract over the litellm transport ---

def test_llm_backend_prompt_json_parses(patch_litellm):
    fake = patch_litellm('{"order": ["x", "y"]}')
    b = make_backend("gpt-5.4")
    out = b.prompt_json("rank these", schemas.RANKING, tag="ranker")
    assert out == {"order": ["x", "y"]}
    # the call reached litellm with the right model + drop_params
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "gpt-5.4"
    assert fake.calls[0]["drop_params"] is True


def test_llm_backend_token_usage_and_cost(patch_litellm):
    patch_litellm('{"order": []}')
    b = make_backend("gpt-5.4")
    b.prompt_json("rank", schemas.RANKING, tag="ranker")
    usage = b.token_usage()
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["cost_usd"] == pytest.approx(0.0012)


def test_llm_backend_cost_defaults_to_zero_on_error(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "completion", _fake_completion('{"order": []}'))

    def _boom(**kw):
        raise RuntimeError("no pricing for this model")

    monkeypatch.setattr(litellm, "completion_cost", _boom)
    b = make_backend("some/unknown-model")
    b.prompt_json("rank", schemas.RANKING, tag="ranker")
    assert b.token_usage()["cost_usd"] == 0.0
