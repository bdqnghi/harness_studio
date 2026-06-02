"""Unit tests for the M2 strategy-unit helpers (diagnoser/reviewer/ranker) and
the family-label derivation."""

from studio.backends.mock import MockBackend
from studio.components import diagnoser, ranker, reviewer, strategist
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


def test_reviewer_keeps_unlisted_and_respects_drop():
    summaries = [{"strategy_id": "a", "family_label": "x", "changed_parts": [], "intent": ""}]
    be = MockBackend(json_responses={"reviewer": [{"keep": [], "drop": []}]})
    verdict = reviewer.review(be, summaries, do_not_repeat=[])
    assert verdict == {"keep": [], "drop": []}


def test_reviewer_empty_input_no_call():
    be = MockBackend()
    assert reviewer.review(be, [], do_not_repeat=[]) == {"keep": [], "drop": []}


def test_ranker_single_strategy_no_call():
    be = MockBackend()
    assert ranker.rank(be, [{"strategy_id": "only"}]) == ["only"]


def test_ranker_appends_missing_ids():
    summaries = [{"strategy_id": "a"}, {"strategy_id": "b"}, {"strategy_id": "c"}]
    # ranker drops "c"; rank() must append it so it still gets tested
    be = MockBackend(json_responses={"ranker": [{"order": ["b", "a"]}]})
    assert ranker.rank(be, summaries) == ["b", "a", "c"]


def test_family_label_from_changed_parts():
    assert strategist.family_label({PartType.TOOL_CODE: ["t.py"]}) == "tool_code"
    label = strategist.family_label(
        {PartType.INSTRUCTIONS: ["i.txt"], PartType.TOOL_CODE: ["t.py"]}
    )
    assert label == "instructions+tool_code"
    assert strategist.family_label({}) == "none"


def test_diversification_hints_distinct_and_padded():
    diag = [{"blamed_part": "tool_code"}, {"blamed_part": "instructions"}]
    hints = strategist.diversification_hints(diag, 3)
    assert len(hints) == 3
