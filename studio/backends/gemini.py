"""Programmatic Gemini backend — our own agentic loop, no CLI subprocess.

This replaces ``claude -p`` / ``gemini -p`` as the proposer. It implements the
Backend ABC with direct Gemini API calls, so it scales (an in-process,
connection-pooled, thread-safe client instead of a cold subprocess per call) and
the rest of the optimizer (orchestrator + every component) is untouched.

* ``prompt_json`` (Tier B) — one completion → schema-validated JSON, retry-once
  on malformed output (same contract as the CLI backend).
* ``run_agent`` (Tier A) — a multi-turn, OpenAI-style **tool-calling loop** that
  reads and edits files in a *jailed* candidate workspace, then reports
  ``files_changed`` via the shared snapshot diff. The tool set (read/write/edit/
  list/grep/bash/complete) is borrowed from AHE's evolve agent, trimmed to the
  harness-edit task; the edit semantics and budgeted-reflect framing are
  borrowed from SkillOpt.

Routing is via Gemini's OpenAI-compatible endpoint, so this optimizer and AHE's
evolve agent both propose with the *same* model (Gemini 3.5 Flash) — isolating
the optimization *algorithm* in the head-to-head.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from .. import schemas
from .base import AgentResult, Backend
from ._fsdiff import diff, snapshot
from ._jsonio import JSONParseError, extract_json

DEFAULT_BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_MODEL = os.environ.get("STUDIO_GEMINI_MODEL", "gemini-3.5-flash")
DEFAULT_TIER_A_MODEL = os.environ.get("STUDIO_TIER_A_MODEL", DEFAULT_MODEL)
DEFAULT_TIER_B_MODEL = os.environ.get("STUDIO_TIER_B_MODEL", DEFAULT_MODEL)

MAX_TOOL_OUTPUT = 10_000   # cap each tool result fed back to the model (context budget)
MAX_READ_CHARS = 16_000    # cap a single read_file result
DEFAULT_MAX_TURNS = int(os.environ.get("STUDIO_TIER_A_MAX_TURNS", "40"))
DEFAULT_MAX_TOKENS = 16_384

# Rough Gemini-Flash pricing ($/token) — informational only; the head-to-head's
# cost metric is task-runs, not USD. Override via env if needed.
_PRICE_IN = float(os.environ.get("STUDIO_GEMINI_PRICE_IN", "0.30")) / 1e6
_PRICE_OUT = float(os.environ.get("STUDIO_GEMINI_PRICE_OUT", "2.50")) / 1e6


class GeminiBackendError(RuntimeError):
    pass


class ToolError(Exception):
    """A recoverable tool failure — its message is fed back to the model."""


# --- tool schemas (OpenAI function-calling format) ---

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the workspace. Returns line-numbered content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "path relative to the workspace"},
            "offset": {"type": "integer", "description": "1-based first line to read (optional)"},
            "limit": {"type": "integer", "description": "max lines to read (optional)"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file in the workspace (creates parent dirs).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact substring in a workspace file. `old` must occur exactly once unless replace_all=true.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "exact text to find"},
            "new": {"type": "string", "description": "replacement text"},
            "replace_all": {"type": "boolean"},
        }, "required": ["path", "old", "new"]},
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List entries under a directory in the workspace (recursive, files marked).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "dir relative to workspace; '.' for root"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search workspace files for a regex; returns matching path:line: text.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "subdir to search (optional)"},
            "context": {"type": "integer", "description": "lines of context around each match (optional)"},
            "head_limit": {"type": "integer", "description": "max matches to return (default 200)"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "complete_task",
        "description": "Call this when your edits are written and you are done. Provide a one-paragraph summary.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
        }, "required": ["summary"]},
    }},
]

SYSTEM_PREAMBLE = """You are a coding agent that improves an AI agent "harness" — a directory of prompt, config, and tool files for an autonomous coding agent. You edit files directly with the provided tools, then call complete_task.

Workspace (the ONLY directory you may modify): {workspace}
Additional read-only directories: {read_dirs}

Rules:
- First use read_file / list_dir / grep to understand the harness before changing it.
- Implement the requested strategy as the smallest coherent change that could plausibly fix the failing tasks.
- Keep the harness valid and bootable: never break YAML/JSON/Python syntax. If you add a tool/middleware, register it where the harness expects.
- Only edit files inside the workspace, and never edit any file listed as do-not-touch in the instruction.
- When your edits are written, call complete_task with a one-paragraph summary of what you changed and why. Do not call complete_task before writing your edits."""


# Read-only "explore" loop (the localizer's Explore-subagent analog). Same tool
# loop as run_agent but NO write/edit tools, and it terminates by calling
# submit_findings with a JSON conclusion validated against a caller schema.
_READONLY_TOOL_NAMES = {"read_file", "list_dir", "grep"}
SUBMIT_FINDINGS_SCHEMA = {"type": "function", "function": {
    "name": "submit_findings",
    "description": "Call this once you have located the cause. Provide your conclusion "
                   "as JSON in `findings`, matching the schema the caller requested. "
                   "Every quote/current_text you cite MUST be copied verbatim from a "
                   "file you actually read — do not paraphrase or invent.",
    "parameters": {"type": "object", "properties": {
        "findings": {"type": "object", "description": "the JSON conclusion"},
    }, "required": ["findings"]},
}}
READONLY_TOOL_SCHEMAS = [t for t in TOOL_SCHEMAS
                         if t["function"]["name"] in _READONLY_TOOL_NAMES] + [SUBMIT_FINDINGS_SCHEMA]

EXPLORE_PREAMBLE = """You are a localization specialist for an AI-agent harness optimizer. You investigate WHY benchmark tasks failed and pinpoint exactly what to change.

Read-only directories you may inspect: {read_dirs}

You CANNOT modify anything — you have only read_file, list_dir, and grep. Your job:
- Read the failure evidence (the verifier's failed checks + the causal transcript windows) and the editable harness files.
- Localize the cause to a specific span/rule of a specific editable file — not just "the instructions".
- Cite your evidence: quote the exact transcript text and the exact current harness text you would change (copied verbatim from what you read).
- Then call submit_findings with the JSON conclusion. Be fast and use parallel tool calls where possible."""


def _cap(s: str, n: int) -> str:
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[:n] + f"\n... [truncated {len(s) - n} chars]"


def _assistant_dict(msg) -> dict:
    """Serialize an assistant message back into the transcript, preserving
    tool_calls and Gemini's per-call ``extra_content`` (thought_signature) so
    multi-turn reasoning stays coherent over the OpenAI-compat endpoint."""
    d: dict = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        calls = []
        for tc in msg.tool_calls:
            call = {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            extra = getattr(tc, "model_extra", None) or {}
            if extra.get("extra_content") is not None:
                call["extra_content"] = extra["extra_content"]
            calls.append(call)
        d["tool_calls"] = calls
    return d


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(
        self,
        *,
        tier_a_model: str = DEFAULT_TIER_A_MODEL,
        tier_b_model: str = DEFAULT_TIER_B_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        api_style: str = "gemini",
        max_turns: int = DEFAULT_MAX_TURNS,
        max_retries: int = 5,
        log_dir: Path | None = None,
        client=None,
    ) -> None:
        self.tier_a_model = tier_a_model
        self.tier_b_model = tier_b_model
        self.max_turns = max_turns
        self.max_retries = max_retries
        # OpenAI reasoning models (gpt-5.x) reject ``max_tokens`` and want
        # ``max_completion_tokens``; the Gemini OpenAI-compat endpoint takes
        # ``max_tokens``. Same OpenAI SDK, one parameter name differs.
        self.api_style = api_style
        self._tokens_param = (
            "max_completion_tokens" if api_style == "openai" else "max_tokens"
        )
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            env_key = "OPENAI_API_KEY" if api_style == "openai" else "GEMINI_API_KEY"
            key = api_key or os.environ.get(env_key)
            if not key:
                raise GeminiBackendError(f"{env_key} is not set")
            self._client = OpenAI(api_key=key, base_url=base_url)

    # --- usage / cost accounting (thread-safe) ---

    def _track(self, usage) -> None:
        if not usage:
            return
        with self._lock:
            self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def _toks(self) -> tuple[int, int]:
        with self._lock:
            return self._prompt_tokens, self._completion_tokens

    def token_usage(self) -> dict:
        p, c = self._toks()
        return {"prompt_tokens": p, "completion_tokens": c,
                "cost_usd": round(p * _PRICE_IN + c * _PRICE_OUT, 4)}

    # --- low-level completion with retry + thinking-model guard ---

    def _is_retryable(self, e) -> bool:
        try:
            import openai
            if isinstance(e, (openai.RateLimitError, openai.APITimeoutError,
                              openai.APIConnectionError, openai.InternalServerError)):
                return True
        except Exception:
            pass
        return getattr(e, "status_code", None) in (429, 500, 502, 503, 504)

    def _complete(self, messages, *, model, tools=None, tag="", max_tokens=DEFAULT_MAX_TOKENS):
        last = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {"model": model, "messages": messages,
                          self._tokens_param: max_tokens}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — classify then re-raise
                last = e
                if self._is_retryable(e) and attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise GeminiBackendError(f"Gemini API error (tag={tag}): {e}") from e
            self._track(getattr(resp, "usage", None))
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
        raise GeminiBackendError(f"Gemini completion failed (tag={tag}): {last}")

    def _log(self, tag, resp) -> None:
        if not self.log_dir:
            return
        try:
            n = len(list(self.log_dir.glob("*.json")))
            (self.log_dir / f"{n:04d}_{tag or 'call'}.json").write_text(
                resp.model_dump_json(indent=2) if hasattr(resp, "model_dump_json") else str(resp)
            )
        except Exception:
            pass

    # --- Tier B: bounded prompt -> validated JSON ---

    def prompt_json(self, prompt, schema, *, tag="", model=None):
        base = (
            f"{prompt}\n\nReturn ONLY a JSON value conforming to this JSON Schema, "
            f"with no prose and no code fences:\n{json.dumps(schema)}"
        )
        full, last_err = base, None
        for _ in range(2):
            resp = self._complete(
                [{"role": "user", "content": full}],
                model=model or self.tier_b_model, tag=tag,
            )
            text = resp.choices[0].message.content or ""
            try:
                data = extract_json(text)
                schemas.validate(data, schema)
                return data
            except (JSONParseError, schemas.SchemaError) as e:
                last_err = e
                full = f"{base}\n\nYour previous reply was invalid ({e}). Reply with valid JSON only."
        raise GeminiBackendError(f"invalid JSON after retry (tag={tag}): {last_err}")

    # --- Tier A: agentic tool-calling loop that edits the workspace ---

    def run_agent(self, instruction, *, workspace, skill="", tag="",
                  model=None, read_dirs=None, timeout=1800):
        workspace = Path(workspace).resolve()
        roots = [workspace] + [Path(d).resolve() for d in (read_dirs or [])]
        before = snapshot(workspace)
        tok0 = self._toks()

        sys_text = SYSTEM_PREAMBLE.format(
            workspace=workspace,
            read_dirs=", ".join(str(d) for d in roots[1:]) or "(none)",
        )
        if skill:
            sys_text += "\n\n## Task-specific guidance\n" + skill
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": instruction},
        ]

        deadline = time.monotonic() + timeout
        final_text, done, turns = "", False, 0
        for turn in range(self.max_turns):
            turns = turn + 1
            if time.monotonic() > deadline:
                final_text = final_text or "(stopped: timeout)"
                break
            resp = self._complete(messages, model=model or self.tier_a_model,
                                  tools=TOOL_SCHEMAS, tag=tag)
            msg = resp.choices[0].message
            messages.append(_assistant_dict(msg))
            if not msg.tool_calls:
                final_text = msg.content or final_text
                break
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "complete_task":
                    final_text = args.get("summary", "") or final_text
                    done = True
                    result = "Task marked complete."
                else:
                    try:
                        result = self._dispatch(name, args, workspace, roots)
                    except ToolError as e:
                        result = f"ERROR: {e}"
                    except Exception as e:  # noqa: BLE001
                        result = f"ERROR: {type(e).__name__}: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": _cap(result, MAX_TOOL_OUTPUT)})
            if done:
                break

        after = snapshot(workspace)
        dp = self._toks()[0] - tok0[0]
        dc = self._toks()[1] - tok0[1]
        return AgentResult(
            text=final_text,
            files_changed=diff(before, after),
            cost_usd=round(dp * _PRICE_IN + dc * _PRICE_OUT, 4),
            raw={"turns": turns, "prompt_tokens": dp, "completion_tokens": dc, "tag": tag},
        )

    # --- Tier B (agentic): read-only exploration that returns a JSON conclusion ---

    def run_explore(self, instruction, *, read_dirs, schema, tag="",
                    model=None, max_turns=None):
        from .. import schemas as _schemas

        roots = [Path(d).resolve() for d in (read_dirs or [])]
        if not roots:
            raise GeminiBackendError("run_explore requires at least one read_dir")
        messages = [
            {"role": "system",
             "content": EXPLORE_PREAMBLE.format(read_dirs=", ".join(str(d) for d in roots))},
            {"role": "user", "content": instruction},
        ]
        limit = max_turns or self.max_turns
        last_err = None
        for _turn in range(limit):
            resp = self._complete(messages, model=model or self.tier_b_model,
                                  tools=READONLY_TOOL_SCHEMAS, tag=tag)
            msg = resp.choices[0].message
            messages.append(_assistant_dict(msg))
            if not msg.tool_calls:
                messages.append({"role": "user",
                                 "content": "Call submit_findings with your JSON conclusion."})
                continue
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                if name == "submit_findings":
                    payload = args.get("findings", args)
                    try:
                        _schemas.validate(payload, schema)
                        return payload
                    except _schemas.SchemaError as e:
                        last_err = e
                        result = (f"ERROR: findings did not match the schema: {e}. "
                                  "Fix and call submit_findings again.")
                else:
                    try:
                        result = self._dispatch_readonly(name, args, roots)
                    except ToolError as e:
                        result = f"ERROR: {e}"
                    except Exception as e:  # noqa: BLE001
                        result = f"ERROR: {type(e).__name__}: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": _cap(result, MAX_TOOL_OUTPUT)})
        raise GeminiBackendError(
            f"run_explore did not submit valid findings (tag={tag}): {last_err}")

    def _dispatch_readonly(self, name, args, roots) -> str:
        if name == "read_file":
            return self._read_file(args, roots)
        if name == "list_dir":
            return self._list_dir(args, roots)
        if name == "grep":
            return self._grep(args, roots)
        raise ToolError(f"tool {name!r} is not available in read-only mode")

    # --- tool dispatch (workspace-jailed) ---

    def _dispatch(self, name, args, workspace, roots) -> str:
        if name == "read_file":
            return self._read_file(args, roots)
        if name == "write_file":
            return self._write_file(args, workspace)
        if name == "edit_file":
            return self._edit_file(args, workspace)
        if name == "list_dir":
            return self._list_dir(args, roots)
        if name == "grep":
            return self._grep(args, roots)
        raise ToolError(f"unknown tool {name!r}")

    @staticmethod
    def _resolve(roots, path, *, write=False) -> Path:
        p = Path(path)
        cand = (p if p.is_absolute() else (roots[0] / p)).resolve()
        allowed = [roots[0]] if write else roots
        for r in allowed:
            if cand == r or r in cand.parents:
                return cand
        where = "workspace" if write else "allowed roots"
        raise ToolError(f"path {path!r} is outside the {where}")

    def _read_file(self, args, roots) -> str:
        target = self._resolve(roots, args["path"])
        if not target.is_file():
            raise ToolError(f"no such file: {args['path']}")
        lines = target.read_text(errors="replace").splitlines()
        offset = max(1, int(args.get("offset", 1)))
        limit = int(args.get("limit", len(lines)))
        chunk = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(f"{offset + i}\t{ln}" for i, ln in enumerate(chunk))
        return _cap(body, MAX_READ_CHARS) if body else "(empty file)"

    def _write_file(self, args, workspace) -> str:
        target = self._resolve([workspace], args["path"], write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"])
        return f"wrote {args['path']} ({len(args['content'])} chars)"

    def _edit_file(self, args, workspace) -> str:
        target = self._resolve([workspace], args["path"], write=True)
        if not target.is_file():
            raise ToolError(f"no such file: {args['path']}")
        content = target.read_text(errors="replace")
        old, new = args["old"], args["new"]
        if old not in content:
            raise ToolError(f"`old` text not found in {args['path']}")
        count = content.count(old)
        if count > 1 and not args.get("replace_all"):
            raise ToolError(f"`old` occurs {count}x in {args['path']}; pass replace_all=true or give more context")
        target.write_text(content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1))
        return f"edited {args['path']} ({count if args.get('replace_all') else 1} replacement(s))"

    def _list_dir(self, args, roots) -> str:
        base = self._resolve(roots, args.get("path", ".") or ".")
        if not base.is_dir():
            raise ToolError(f"not a directory: {args.get('path')}")
        out = []
        for p in sorted(base.rglob("*")):
            if any(x in p.parts for x in ("__pycache__", ".git", ".pytest_cache")):
                continue
            rel = p.relative_to(base)
            out.append(f"{rel}/" if p.is_dir() else str(rel))
        return "\n".join(out[:400]) or "(empty)"

    def _grep(self, args, roots) -> str:
        import re as _re
        base = self._resolve(roots, args.get("path", ".") or ".")
        try:
            rx = _re.compile(args["pattern"])
        except _re.error as e:
            raise ToolError(f"bad regex: {e}")
        # Optional ripgrep-style context lines (-C) and a head_limit, so the
        # explorer can pull the surrounding rule/code around a match.
        context = max(0, int(args.get("context", 0) or 0))
        head_limit = int(args.get("head_limit", 200) or 200)
        hits, root = [], (base if base.is_dir() else base.parent)
        files = base.rglob("*") if base.is_dir() else [base]
        for p in sorted(files):
            if not p.is_file() or any(x in p.parts for x in ("__pycache__", ".git", ".pytest_cache")):
                continue
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError:
                continue
            rel = p.relative_to(root)
            for i, ln in enumerate(lines, 1):
                if rx.search(ln):
                    if context:
                        lo, hi = max(1, i - context), min(len(lines), i + context)
                        block = "\n".join(f"{rel}:{j}: {lines[j - 1].rstrip()[:200]}"
                                          for j in range(lo, hi + 1))
                        hits.append(block)
                    else:
                        hits.append(f"{rel}:{i}: {ln.strip()[:200]}")
                    if len(hits) >= head_limit:
                        return "\n".join(hits) + "\n... [more matches truncated]"
        return "\n".join(hits) or "(no matches)"
