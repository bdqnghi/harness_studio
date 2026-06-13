"""AI-helper backends (the Backend seam)."""

from __future__ import annotations

from .base import AgentResult, Backend
from .mock import MockBackend

__all__ = ["AgentResult", "Backend", "MockBackend", "LLMBackend", "LLMLoopBackend"]


def __getattr__(name):
    # Lazy so importing the package never pulls in litellm unless a real backend
    # is actually constructed.
    if name == "LLMLoopBackend":
        from .llm_loop import LLMLoopBackend

        return LLMLoopBackend
    if name == "LLMBackend":
        from .llm import LLMBackend

        return LLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
