"""A deterministic, network-free Backend for tests.

Tier-B calls return scripted JSON; Tier-A calls run scripted *actions* that
mutate the candidate workspace on disk. Responses/actions are keyed by ``tag``
(the calling helper) and consumed FIFO, so a test can lay out an exact sequence
of proposer behaviors — a mix of good edits, no-op edits, regressions, and
non-booting edits — to exercise every branch of the acceptance and structural check.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from .base import AgentResult, Backend

# A scripted Tier-A action edits files under the given workspace root. It may
# optionally take a second argument (the instruction text) — useful for testing
# proposer behavior that depends on the family map / pivot directives.
AgentAction = Callable[..., None]


class MockBackend(Backend):
    name = "mock"

    def __init__(
        self,
        json_responses: dict[str, list] | None = None,
        agent_actions: dict[str, list[AgentAction]] | None = None,
        explore_responses: dict[str, list] | None = None,
    ) -> None:
        # Copy so callers can reuse their script dicts across backends.
        self._json = {k: list(v) for k, v in (json_responses or {}).items()}
        self._actions = {k: list(v) for k, v in (agent_actions or {}).items()}
        self._explore = {k: list(v) for k, v in (explore_responses or {}).items()}
        self.calls: list[tuple[str, str]] = []  # (kind, tag) call log
        self.prompt_log: list[tuple[str, str]] = []  # (tag, prompt/instruction text)

    def prompt_json(self, prompt, schema, *, tag="", model=None):
        self.calls.append(("prompt_json", tag))
        self.prompt_log.append((tag, prompt))
        queue = self._json.get(tag)
        if not queue:
            raise AssertionError(f"MockBackend: no scripted JSON for tag {tag!r}")
        return queue.pop(0)

    def run_explore(self, instruction, *, read_dirs, schema, tag="",
                    model=None, max_turns=None):
        self.calls.append(("run_explore", tag))
        self.prompt_log.append((tag, instruction))
        queue = self._explore.get(tag)
        if not queue:
            raise AssertionError(f"MockBackend: no scripted explore response for tag {tag!r}")
        return queue.pop(0)

    def run_agent(
        self,
        instruction,
        *,
        workspace,
        skill="",
        tag="",
        model=None,
        read_dirs=None,
        timeout=1800,
    ):
        self.calls.append(("run_agent", tag))
        self.prompt_log.append((tag, instruction))
        queue = self._actions.get(tag)
        if not queue:
            raise AssertionError(f"MockBackend: no scripted action for tag {tag!r}")
        action = queue.pop(0)
        workspace = Path(workspace)
        before = _snapshot(workspace)
        # Pass the instruction too if the action accepts a second argument.
        if len(inspect.signature(action).parameters) >= 2:
            action(workspace, instruction)
        else:
            action(workspace)
        after = _snapshot(workspace)
        return AgentResult(text="mock-agent", files_changed=_diff(before, after))


_IGNORE = {"__pycache__", ".git", ".pytest_cache"}


def _snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(part in _IGNORE for part in p.parts):
            out[str(p.relative_to(root))] = p.read_text(errors="replace")
    return out


def _diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Relative paths that were added, removed, or modified."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
