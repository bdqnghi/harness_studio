"""Unit tests for GeminiBackend against a stubbed client — no network.

Exercises the contract the orchestrator depends on: Tier-B JSON (retry on
malformed, retry on schema-violation, fail after two), the Tier-A tool loop
(edit/complete, workspace jail, files_changed diff, max-turns stop), and the
thinking-model empty-output guard.
"""

from __future__ import annotations

import json

import pytest

from studio import schemas
from studio.backends.gemini import GeminiBackend, GeminiBackendError, TOOL_SCHEMAS


# --- minimal fake of the OpenAI chat-completions client ---

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, idx, name, args, extra=None):
        self.id = f"call_{idx}"
        self.type = "function"
        self.function = _Fn(name, json.dumps(args))
        self._extra = extra or {}

    @property
    def model_extra(self):
        return self._extra


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Resp:
    def __init__(self, message, finish_reason):
        self.choices = [type("C", (), {"message": message, "finish_reason": finish_reason})()]
        self.usage = _Usage()

    def model_dump_json(self, indent=None):
        return "{}"


def _text(content, finish="stop"):
    return _Resp(_Msg(content=content), finish)


def _tools(calls, finish="tool_calls"):
    tcs = [_TC(i, c["name"], c.get("args", {}), c.get("extra")) for i, c in enumerate(calls)]
    return _Resp(_Msg(tool_calls=tcs), finish)


class _Completions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.script, "fake client ran out of scripted responses"
        return self.script.pop(0)


class FakeClient:
    def __init__(self, script):
        self.chat = type("Chat", (), {"completions": _Completions(script)})()


def _backend(script, **kw):
    return GeminiBackend(client=FakeClient(script), **kw)


# --- Tier B: prompt_json ---

def test_prompt_json_retries_on_malformed():
    b = _backend([_text("not json at all"), _text('{"order": ["a", "b"]}')])
    out = b.prompt_json("rank these", schemas.RANKING, tag="ranker")
    assert out == {"order": ["a", "b"]}
    assert len(b._client.chat.completions.calls) == 2


def test_prompt_json_retries_on_schema_violation():
    # first parses as JSON but violates schema (missing 'order'), second is valid
    b = _backend([_text('{"wrong": 1}'), _text('{"order": []}')])
    out = b.prompt_json("rank", schemas.RANKING, tag="ranker")
    assert out == {"order": []}


def test_prompt_json_raises_after_two_failures():
    b = _backend([_text("nope"), _text("still nope")])
    with pytest.raises(GeminiBackendError):
        b.prompt_json("rank", schemas.RANKING, tag="ranker")


def test_prompt_json_recovers_fenced_json():
    b = _backend([_text('```json\n{"order": ["x"]}\n```')])
    assert b.prompt_json("rank", schemas.RANKING, tag="ranker") == {"order": ["x"]}


# --- thinking-model guard (empty output, finish_reason=length) ---

def test_complete_recovers_from_empty_length_output():
    # first response: thinking ate the whole budget -> empty, finish_reason=length
    b = _backend([_text("", finish="length"), _text('{"order": []}')])
    out = b.prompt_json("rank", schemas.RANKING, tag="ranker")
    assert out == {"order": []}
    assert len(b._client.chat.completions.calls) == 2  # guard re-issued the call


# --- Tier A: run_agent tool loop ---

def test_run_agent_edits_then_completes(tmp_path):
    (tmp_path / "a.txt").write_text("foo bar\n")
    b = _backend([
        _tools([{"name": "edit_file", "args": {"path": "a.txt", "old": "foo", "new": "baz"}}]),
        _tools([{"name": "complete_task", "args": {"summary": "changed foo->baz"}}]),
    ])
    res = b.run_agent("improve it", workspace=tmp_path, tag="strategist")
    assert (tmp_path / "a.txt").read_text() == "baz bar\n"
    assert res.files_changed == ["a.txt"]
    assert res.text == "changed foo->baz"
    assert res.raw["turns"] == 2


def test_run_agent_write_then_read_roundtrip(tmp_path):
    b = _backend([
        _tools([{"name": "write_file", "args": {"path": "sub/new.py", "content": "x = 1\n"}}]),
        _tools([{"name": "read_file", "args": {"path": "sub/new.py"}}]),
        _tools([{"name": "complete_task", "args": {"summary": "added file"}}]),
    ])
    res = b.run_agent("add a file", workspace=tmp_path, tag="strategist")
    assert (tmp_path / "sub" / "new.py").read_text() == "x = 1\n"
    assert "sub/new.py" in res.files_changed


def test_run_agent_workspace_jail_blocks_escape(tmp_path):
    outside = tmp_path.parent / "escape.txt"
    b = _backend([
        _tools([{"name": "write_file", "args": {"path": "../escape.txt", "content": "pwned"}}]),
        _tools([{"name": "complete_task", "args": {"summary": "tried to escape"}}]),
    ])
    res = b.run_agent("do it", workspace=tmp_path, tag="strategist")
    assert not outside.exists()          # jail held
    assert res.files_changed == []       # nothing changed in workspace


def test_run_agent_edit_missing_old_is_recoverable(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    b = _backend([
        _tools([{"name": "edit_file", "args": {"path": "a.txt", "old": "ABSENT", "new": "x"}}]),
        _tools([{"name": "complete_task", "args": {"summary": "gave up"}}]),
    ])
    res = b.run_agent("edit", workspace=tmp_path, tag="strategist")
    assert (tmp_path / "a.txt").read_text() == "hello\n"  # unchanged, no crash
    assert res.files_changed == []


def test_run_agent_stops_at_max_turns(tmp_path):
    (tmp_path / "a.txt").write_text("hi\n")
    # never calls complete_task — keep reading
    script = [_tools([{"name": "read_file", "args": {"path": "a.txt"}}]) for _ in range(3)]
    b = _backend(script, max_turns=3)
    res = b.run_agent("loop", workspace=tmp_path, tag="strategist")
    assert res.raw["turns"] == 3
    assert len(b._client.chat.completions.calls) == 3


def test_run_agent_does_not_expose_shell():
    names = {tool["function"]["name"] for tool in TOOL_SCHEMAS}
    assert "run_bash" not in names


def test_run_agent_preserves_thought_signature_in_transcript(tmp_path):
    (tmp_path / "a.txt").write_text("foo\n")
    extra = {"extra_content": {"google": {"thought_signature": "SIG123"}}}
    b = _backend([
        _tools([{"name": "edit_file", "args": {"path": "a.txt", "old": "foo", "new": "bar"}, "extra": extra}]),
        _tools([{"name": "complete_task", "args": {"summary": "done"}}]),
    ])
    b.run_agent("edit", workspace=tmp_path, tag="strategist")
    # the 2nd create call must carry the assistant message with the signature
    second_call_messages = b._client.chat.completions.calls[1]["messages"]
    assistant = [m for m in second_call_messages if m.get("role") == "assistant"][0]
    assert assistant["tool_calls"][0]["extra_content"] == extra["extra_content"]
