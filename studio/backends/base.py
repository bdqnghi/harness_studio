"""The AI-helper seam.

Every AI helper (PRD §1.6) runs through a Backend. There are two call shapes:

* ``prompt_json`` — Tier B: a bounded prompt in, validated JSON out. Used by the
  Mapper, Diagnoser, Reviewer, Ranker.
* ``run_agent`` — Tier A: a filesystem-navigating coding agent that reads a
  workspace and *edits files in place*. Used by the Strategist and Meta-agent.

Putting both behind one interface lets the whole pipeline run against a scripted
``MockBackend`` (deterministic, free, used by every unit/integration test) or the
real ``ClaudeCLIBackend`` (subprocess ``claude -p``) with no change to the loop.

Trust boundary (PRD §3): a Backend is never handed to the gate. AI helpers only
ever propose; only code decides.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentResult:
    """Outcome of a Tier-A coding-agent run."""

    text: str = ""
    files_changed: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)


class Backend(abc.ABC):
    """Common interface for Tier-A and Tier-B AI helpers."""

    name: str = "backend"

    @abc.abstractmethod
    def prompt_json(
        self,
        prompt: str,
        schema: dict,
        *,
        tag: str = "",
        model: str | None = None,
    ) -> dict | list:
        """Tier B: run a bounded prompt and return JSON conforming to ``schema``.

        ``tag`` labels the calling helper (used for logging by the real backend
        and as a script key by the mock)."""

    @abc.abstractmethod
    def run_agent(
        self,
        instruction: str,
        *,
        workspace: Path,
        skill: str = "",
        tag: str = "",
        model: str | None = None,
        read_dirs: list[Path] | None = None,
        timeout: int = 1800,
    ) -> AgentResult:
        """Tier A: run a coding agent in ``workspace`` that may edit files there.

        ``skill`` is the minimal SKILL.md text steering the agent (workspace
        layout, do-not-touch list, output contract). ``read_dirs`` are extra
        read-only directories (e.g. prior traces) the agent may inspect."""
