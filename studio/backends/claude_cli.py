"""The real AI-helper backend: subprocess ``claude -p``.

Tier B uses ``--output-format json --json-schema`` for structured output. Tier A
runs Claude Code as a coding agent with file tools in the candidate workspace —
the proposer pattern from meta-harness (Claude Code as a coding agent, not a raw
LLM, because it must selectively inspect history and validate edits by editing
files directly). The agent is steered only by a minimal skill via
``--append-system-prompt`` (meta-harness's lesson: "write a good skill").
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .. import schemas
from .base import AgentResult, Backend

DEFAULT_TIER_B_MODEL = os.environ.get("STUDIO_TIER_B_MODEL", "claude-haiku-4-5-20251001")
DEFAULT_TIER_A_MODEL = os.environ.get("STUDIO_TIER_A_MODEL", "claude-sonnet-4-6")
TIER_A_TOOLS = ["Read", "Edit", "Write", "Grep", "Glob"]


class ClaudeCLIError(RuntimeError):
    pass


class ClaudeCLIBackend(Backend):
    name = "claude"

    def __init__(
        self,
        *,
        tier_b_model: str = DEFAULT_TIER_B_MODEL,
        tier_a_model: str = DEFAULT_TIER_A_MODEL,
        log_dir: Path | None = None,
    ) -> None:
        self.tier_b_model = tier_b_model
        self.tier_a_model = tier_a_model
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    # --- Tier B: bounded prompt -> validated JSON ---

    def prompt_json(self, prompt, schema, *, tag="", model=None):
        # The CLI's --json-schema flag is unreliable on some builds, so we steer
        # the model with the schema in-prompt and validate the parsed result
        # ourselves, retrying once on malformed output (PRD §5.12).
        base = (
            f"{prompt}\n\nReturn ONLY a JSON value conforming to this JSON Schema, "
            f"with no prose and no code fences:\n{json.dumps(schema)}"
        )
        full_prompt, last_err = base, None
        for _ in range(2):
            cmd = [
                "claude", "-p", full_prompt,
                "--output-format", "json",
                "--model", model or self.tier_b_model,
            ]
            envelope = self._run(cmd, tag=tag, timeout=600)
            try:
                data = _extract_json(str(envelope.get("result", "")))
                schemas.validate(data, schema)
                return data
            except (ClaudeCLIError, schemas.SchemaError) as e:
                last_err = e
                full_prompt = f"{base}\n\nYour previous reply was invalid ({e}). Reply with valid JSON only."
        raise ClaudeCLIError(f"invalid JSON after retry (tag={tag}): {last_err}")

    # --- Tier A: coding agent that edits files in the workspace ---

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
        workspace = Path(workspace)
        before = _snapshot(workspace)
        cmd = [
            "claude", "-p", instruction,
            "--output-format", "json",
            "--add-dir", str(workspace),
            "--allowed-tools", " ".join(TIER_A_TOOLS),
            "--model", model or self.tier_a_model,
        ]
        for d in read_dirs or []:
            cmd += ["--add-dir", str(d)]
        if skill:
            cmd += ["--append-system-prompt", skill]
        cmd += ["--permission-mode", "acceptEdits"]

        envelope = self._run(cmd, tag=tag, timeout=timeout, cwd=workspace)
        after = _snapshot(workspace)
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        return AgentResult(
            text=str(envelope.get("result", "")),
            files_changed=changed,
            cost_usd=float(envelope.get("total_cost_usd", 0.0) or 0.0),
            raw=envelope,
        )

    # --- subprocess plumbing ---

    def _run(self, cmd, *, tag, timeout, cwd=None) -> dict:
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL,  # -p reads the prompt from argv, not stdin
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCLIError(f"claude -p timed out after {timeout}s (tag={tag})") from e
        if self.log_dir:
            self._log(tag, cmd, proc)
        if proc.returncode != 0:
            raise ClaudeCLIError(
                f"claude -p failed (tag={tag}, rc={proc.returncode}): {proc.stderr[:2000]}"
            )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIError(
                f"claude -p returned non-JSON (tag={tag}): {proc.stdout[:2000]}"
            ) from e

    def _log(self, tag, cmd, proc) -> None:
        n = len(list(self.log_dir.glob("*.txt")))
        path = self.log_dir / f"{n:04d}_{tag or 'call'}.txt"
        path.write_text(
            f"$ {' '.join(repr(c) if ' ' in c else c for c in cmd)}\n\n"
            f"--- rc={proc.returncode} ---\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
        )


def _extract_json(text: str):
    """Parse a JSON value from model output, tolerating code fences / stray prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first JSON value starting at the first { or [.
    # raw_decode parses one value and ignores any trailing text — O(n), not O(n^2).
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start != -1:
        try:
            return json.JSONDecoder().raw_decode(text, start)[0]
        except json.JSONDecodeError:
            pass
    raise ClaudeCLIError(f"could not parse JSON from model output: {text[:500]}")


def _snapshot(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts:
            try:
                out[str(p.relative_to(root))] = p.read_text(errors="replace")
            except OSError:
                pass
    return out
