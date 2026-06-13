"""Phase 4: the localizer — mode selection, citation guard, fallbacks."""

from __future__ import annotations

import pytest

from studio.backends.mock import MockBackend
from studio.stages.optimize.edit import localizer
from studio.core.evidence import EvidenceStore, TaskEvidence, TraceWindow, VerifierSignal
from studio.core.harness import Harness


def _harness(tmp_path, body="# Policy\n\nBe helpful and resolve the request.\n"):
    root = tmp_path / "h"; root.mkdir()
    h = Harness(root); h.write_file("policy.md", body)
    return h


def _evidence_dir(tmp_path):
    store = EvidenceStore()
    store.put("h1", TaskEvidence(
        task_id="t1", reward=0.0,
        signals=[VerifierSignal("action", "update_reservation", False, "expected update_reservation")],
        windows=[TraceWindow("t1", 0, 2, 3,
                             [{"role": "assistant", "content": "I will book a new flight"}],
                             "action failed")],
    ))
    return store.materialize("h1", tmp_path / "ev")


PATTERNS = [{"pattern_id": "p1", "root_cause": "no rule for changing flights",
             "verifier_cause": "action update_reservation not taken",
             "agent_mechanism": "agent booked new instead of updating",
             "failing_task_ids": ["t1"]}]


def test_choose_mode_single_file_inline_multi_file_agentic(tmp_path):
    ev = _evidence_dir(tmp_path)
    assert localizer.choose_mode("auto", PATTERNS, ["policy.md"], ev) == "inline"
    assert localizer.choose_mode("auto", PATTERNS, ["a.py", "b.py"], ev) == "agentic"
    assert localizer.choose_mode("auto", PATTERNS, ["src/"], ev) == "agentic"  # dir entry
    big = PATTERNS * 3
    assert localizer.choose_mode("auto", big, ["policy.md"], ev) == "agentic"
    # explicit overrides
    assert localizer.choose_mode("inline", big, ["a.py", "b.py"], ev) == "inline"


def test_inline_localize_validates_and_returns(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    good = {"targets": [{
        "pattern_id": "p1", "target_file": "policy.md",
        "current_text": "Be helpful and resolve the request.",   # verbatim in policy.md
        "target_locator": "the conduct section", "change_kind": "add_rule",
        "evidence": [{"task_id": "t1", "quote": "I will book a new flight"}],  # verbatim in corpus
    }]}
    backend = MockBackend(json_responses={"localizer": [good]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="inline")
    assert len(out) == 1 and out[0]["target_file"] == "policy.md"
    assert ("prompt_json", "localizer") in backend.calls


def test_citation_guard_drops_hallucinated_current_text(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    bad = {"targets": [{
        "pattern_id": "p1", "target_file": "policy.md",
        "current_text": "THIS TEXT IS NOT IN THE POLICY AT ALL",   # hallucinated
        "evidence": [{"task_id": "t1", "quote": "I will book a new flight"}],
    }]}
    backend = MockBackend(json_responses={"localizer": [bad]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="inline")
    assert out == []  # dropped — never read-before-act


def test_citation_guard_drops_hallucinated_evidence_quote(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    bad = {"targets": [{
        "pattern_id": "p1", "target_file": "policy.md", "current_text": "",
        "evidence": [{"task_id": "t1", "quote": "the agent did something never in the transcript"}],
    }]}
    backend = MockBackend(json_responses={"localizer": [bad]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="inline")
    assert out == []  # no grounded evidence -> dropped


def test_citation_guard_rejects_non_editable_target(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    bad = {"targets": [{
        "pattern_id": "p1", "target_file": "some_other_file.py", "current_text": "",
        "evidence": [{"task_id": "t1", "quote": "I will book a new flight"}],
    }]}
    backend = MockBackend(json_responses={"localizer": [bad]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="inline")
    assert out == []


def test_agentic_uses_run_explore(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    good = {"targets": [{
        "pattern_id": "p1", "target_file": "policy.md",
        "current_text": "Be helpful and resolve the request.",
        "evidence": [{"task_id": "t1", "quote": "I will book a new flight"}],
    }]}
    backend = MockBackend(explore_responses={"localizer": [good]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["a.py", "b.py"], mode="agentic")
    # policy.md isn't in editable_files here, so it would be dropped — use matching files
    assert out == []  # target_file policy.md not in {a.py,b.py}
    assert ("run_explore", "localizer") in backend.calls


def test_agentic_falls_back_to_inline_when_unsupported(tmp_path):
    """A backend without run_explore (raises NotImplementedError) degrades to inline."""
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    good = {"targets": [{
        "pattern_id": "p1", "target_file": "policy.md",
        "current_text": "Be helpful and resolve the request.",
        "evidence": [{"task_id": "t1", "quote": "I will book a new flight"}],
    }]}
    # explore unscripted -> MockBackend.run_explore raises AssertionError; localize
    # catches and falls back to the scripted inline prompt_json.
    backend = MockBackend(json_responses={"localizer": [good]})
    out = localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="agentic")
    assert len(out) == 1
    assert ("run_explore", "localizer") in backend.calls   # tried agentic first
    assert ("prompt_json", "localizer") in backend.calls   # fell back to inline


def test_localize_empty_when_no_patterns_or_no_editable(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    assert localizer.localize(MockBackend(), [], h, ev, editable_files=["policy.md"]) == []
    assert localizer.localize(MockBackend(), PATTERNS, h, ev, editable_files=[]) == []


def test_localize_backend_error_returns_empty(tmp_path):
    h = _harness(tmp_path)
    ev = _evidence_dir(tmp_path)
    backend = MockBackend()  # no scripted localizer response -> prompt_json raises
    assert localizer.localize(backend, PATTERNS, h, ev, editable_files=["policy.md"], mode="inline") == []
