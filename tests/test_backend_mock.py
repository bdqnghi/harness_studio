import pytest

from studio.backends.mock import MockBackend
from studio.benchmark.toy import build_toy_harness
from studio.benchmark import toy_fixes


def test_prompt_json_fifo():
    be = MockBackend(json_responses={"diagnoser": [{"a": 1}, {"a": 2}]})
    assert be.prompt_json("p", {}, tag="diagnoser") == {"a": 1}
    assert be.prompt_json("p", {}, tag="diagnoser") == {"a": 2}


def test_prompt_json_missing_tag_raises():
    be = MockBackend()
    with pytest.raises(AssertionError):
        be.prompt_json("p", {}, tag="nope")


def test_run_agent_applies_action_and_reports_changes(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    be = MockBackend(agent_actions={"strategist": [toy_fixes.enable_upper]})
    result = be.run_agent("go", workspace=h.root, tag="strategist")
    assert "instructions.txt" in result.files_changed
    assert "ENABLE upper" in h.read_file("instructions.txt")


def test_run_agent_noop_reports_no_changes(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    be = MockBackend(agent_actions={"strategist": [toy_fixes.noop]})
    result = be.run_agent("go", workspace=h.root, tag="strategist")
    assert result.files_changed == []
