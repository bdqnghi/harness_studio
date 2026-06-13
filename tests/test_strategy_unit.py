"""Unit tests for the diagnoser and the family-label derivation."""

from studio.backends.mock import MockBackend
from studio.components import diagnoser, strategist
from studio.components.runner import Failure
from studio.parts import PartType


def test_diagnoser_returns_validated_clusters():
    resp = [{
        "pattern_id": "p1", "description": "x", "root_cause": "timeouts",
        "failing_task_ids": ["t1", "t2"], "blamed_part": "tool_code", "confidence": 0.9,
    }]
    be = MockBackend(json_responses={"diagnoser": [resp]})
    out = diagnoser.diagnose(be, [Failure("t1", "desc"), Failure("t2", "desc")])
    assert out[0]["blamed_part"] == "tool_code"


def test_diagnoser_no_failures_skips_call():
    be = MockBackend()  # no scripted response; must not be called
    assert diagnoser.diagnose(be, []) == []


def test_family_label_from_changed_parts():
    assert strategist.family_label({PartType.TOOL_CODE: ["t.py"]}) == "tool_code"
    label = strategist.family_label(
        {PartType.INSTRUCTIONS: ["i.txt"], PartType.TOOL_CODE: ["t.py"]}
    )
    assert label == "instructions+tool_code"
    assert strategist.family_label({}) == "none"
