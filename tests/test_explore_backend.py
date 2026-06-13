"""Phase 3: read-only `run_explore` loop (the localizer's Explore analog)."""

from __future__ import annotations

import json

import pytest

from studio import schemas
from studio.backends.gemini import GeminiBackend, GeminiBackendError
from studio.backends.mock import MockBackend

# Reuse the fake client harness from the Tier-A/B backend tests.
from tests.test_gemini_backend import FakeClient, _tools


def _backend(script, **kw):
    return GeminiBackend(client=FakeClient(script), **kw)


_GOOD = {"targets": [{"pattern_id": "p1", "target_file": "policy.md",
                      "current_text": "Be helpful.",
                      "evidence": [{"task_id": "t1", "quote": "the agent skipped X"}]}]}


def test_run_explore_returns_validated_findings(tmp_path):
    (tmp_path / "policy.md").write_text("Be helpful.\n")
    (tmp_path / "t1.md").write_text("failed checks: action update_reservation\n")
    b = _backend([
        _tools([{"name": "grep", "args": {"pattern": "update", "path": "."}}]),
        _tools([{"name": "read_file", "args": {"path": "policy.md"}}]),
        _tools([{"name": "submit_findings", "args": {"findings": _GOOD}}]),
    ])
    out = b.run_explore("localize", read_dirs=[tmp_path], schema=schemas.LOCALIZATION, tag="localizer")
    assert out == _GOOD
    assert len(b._client.chat.completions.calls) == 3


def test_run_explore_retries_on_invalid_findings(tmp_path):
    (tmp_path / "policy.md").write_text("Be helpful.\n")
    bad = {"targets": [{"pattern_id": "p1"}]}  # missing required target_file/evidence
    b = _backend([
        _tools([{"name": "submit_findings", "args": {"findings": bad}}]),
        _tools([{"name": "submit_findings", "args": {"findings": _GOOD}}]),
    ])
    out = b.run_explore("localize", read_dirs=[tmp_path], schema=schemas.LOCALIZATION, tag="localizer")
    assert out == _GOOD
    assert len(b._client.chat.completions.calls) == 2  # the bad one was rejected + retried


def test_run_explore_rejects_write_tools(tmp_path):
    (tmp_path / "policy.md").write_text("Be helpful.\n")
    b = _backend([
        _tools([{"name": "edit_file", "args": {"path": "policy.md", "old": "Be", "new": "X"}}]),
        _tools([{"name": "submit_findings", "args": {"findings": _GOOD}}]),
    ])
    out = b.run_explore("localize", read_dirs=[tmp_path], schema=schemas.LOCALIZATION, tag="localizer")
    assert out == _GOOD
    assert (tmp_path / "policy.md").read_text() == "Be helpful.\n"  # NOT modified


def test_run_explore_raises_if_never_submits(tmp_path):
    (tmp_path / "policy.md").write_text("x\n")
    script = [_tools([{"name": "list_dir", "args": {"path": "."}}]) for _ in range(20)]
    b = _backend(script, max_turns=3)
    with pytest.raises(GeminiBackendError):
        b.run_explore("localize", read_dirs=[tmp_path], schema=schemas.LOCALIZATION, tag="localizer")


def test_run_explore_requires_read_dirs(tmp_path):
    b = _backend([])
    with pytest.raises(GeminiBackendError):
        b.run_explore("x", read_dirs=[], schema=schemas.LOCALIZATION, tag="localizer")


def test_grep_context_and_head_limit(tmp_path):
    (tmp_path / "f.md").write_text("a\nb\nMATCH\nd\ne\n")
    b = _backend([])
    out = b._grep({"pattern": "MATCH", "path": ".", "context": 1}, [tmp_path])
    assert "b" in out and "MATCH" in out and "d" in out  # ±1 context lines included


def test_mock_run_explore_is_scripted_and_logged():
    mock = MockBackend(explore_responses={"localizer": [_GOOD]})
    out = mock.run_explore("find it", read_dirs=["/x"], schema=schemas.LOCALIZATION, tag="localizer")
    assert out == _GOOD
    assert ("run_explore", "localizer") in mock.calls
    assert mock.prompt_log[-1] == ("localizer", "find it")


def test_mock_run_explore_unscripted_raises():
    with pytest.raises(AssertionError):
        MockBackend().run_explore("x", read_dirs=["/x"], schema=schemas.LOCALIZATION, tag="localizer")
