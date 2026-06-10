"""Integration test for the two-speed meta-loop (M3).

We plant a plateau: the proposer keeps making a *regressing* edit and the gate
keeps rejecting it. (Under do-no-harm a merely-neutral edit would be accepted,
so the plateau is carried by a real regression.) Only after the Meta-agent writes
a *pivot directive* into the family map (the one mechanism edit at the segment
boundary) does the proposer switch to the fix that actually helps — and the gate
accepts it. This is the AEVO signature: improvement that a fixed-rule search
could not have produced.
"""

from pathlib import Path

from studio.backends.mock import MockBackend
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.benchmark import toy_fixes
from studio.components.family_map import FamilyMap
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator

SPLIT = TaskSplit(
    judging=[f"{f}-{i}" for f in FAMILIES for i in (0, 1)],
    final_exam=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
    audit=[f"{f}-{i}" for f in FAMILIES for i in (4, 5)],
    practice=[f"{f}-{i}" for f in FAMILIES for i in (6, 7, 8, 9, 10, 11)],
)
PIVOT_DIRECTIVE = "fix tool_code reverse op"


def plateau_or_pivot(root: Path, instruction: str) -> None:
    """Behavior depends on the family map handed to the proposer: only once the
    pivot directive mentions reverse does it apply the real fix."""
    tools = (Path(root) / "tools.py").read_text()
    # Only the family-map pivot directive (not the diagnosis) unlocks the real fix.
    if PIVOT_DIRECTIVE in instruction and "BUG" in tools:
        toy_fixes.fix_reverse(root)       # the helpful edit
    else:
        toy_fixes.regress_echo(root)      # a regressing edit -> gate rejects


def meta_pivot(root: Path) -> None:
    """The Meta-agent's single mechanism edit: add a pivot directive."""
    p = Path(root) / "family_map.md"
    fm = FamilyMap.load(p)
    fm.add_pivot(PIVOT_DIRECTIVE)
    fm.save(p)


DIAG = [{
    "pattern_id": "p1", "description": "reverse fails", "root_cause": "buggy reverse",
    "failing_task_ids": ["reverse-0"], "blamed_part": "tool_code", "confidence": 0.6,
}]


def _run(tmp_path):
    backend = MockBackend(
        json_responses={
            "diagnoser": [DIAG] * 6,
            "reviewer": [{"keep": [], "drop": []}] * 6,
        },
        agent_actions={
            "strategist": [plateau_or_pivot] * 4,
            "meta": [meta_pivot],
        },
    )
    config = Config(loop=LoopConfig(
        rounds=4, segment_length=2, wobble_runs=3, strategies_per_round=1,
    ), edits=EditConfig(allow_repair=False))
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=backend, config=config, split=SPLIT, part_map=toy_part_map(),
    )
    return orch.run(), orch


def test_plateau_then_escape_after_mechanism_edit(tmp_path):
    result, orch = _run(tmp_path)
    by_round = {r.round_idx: r for r in result.rounds}

    # Segment 1: the proposer churns; the gate rejects everything (plateau).
    assert not by_round[1].accepted
    assert not by_round[2].accepted

    # The Meta-agent made the one mechanism edit at the boundary.
    assert PIVOT_DIRECTIVE in result.family_map.pivot

    # Segment 2: with the pivot directive, the proposer escapes the plateau.
    assert by_round[3].accepted
    assert result.final_score > result.baseline_final


def test_no_meta_escalation_when_progress(tmp_path):
    # If the first segment already makes progress, the meta-agent must NOT run
    # (no scripted 'meta' action -> would raise if it did).
    backend = MockBackend(
        json_responses={"diagnoser": [DIAG] * 6, "reviewer": [{"keep": [], "drop": []}] * 6},
        agent_actions={"strategist": [toy_fixes.fix_reverse, toy_fixes.enable_upper,
                                      toy_fixes.fix_add_full, toy_fixes.enable_bogus]},
    )
    config = Config(loop=LoopConfig(
        rounds=4, segment_length=2, wobble_runs=3, strategies_per_round=1,
    ), edits=EditConfig(allow_repair=False))
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=backend, config=config, split=SPLIT, part_map=toy_part_map(),
    )
    result = orch.run()  # must not raise (no meta action needed)
    assert result.accepted >= 2
    # A confirmed-working family was promoted by the rule-based update.
    assert result.family_map.works
