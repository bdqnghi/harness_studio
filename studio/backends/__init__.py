"""AI-helper backends (the Backend seam)."""

from __future__ import annotations

from .base import AgentResult, Backend
from .mock import MockBackend

__all__ = ["AgentResult", "Backend", "MockBackend", "ClaudeCLIBackend", "GeminiBackend"]


def __getattr__(name):
    # Lazy so importing the package never pulls in `openai` (Gemini) or requires
    # the `claude` CLI unless that backend is actually used.
    if name == "GeminiBackend":
        from .gemini import GeminiBackend

        return GeminiBackend
    if name == "ClaudeCLIBackend":
        from .claude_cli import ClaudeCLIBackend

        return ClaudeCLIBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
