"""Integration test: the multi-strategy inner loop (M2) climbs the toy.

Each round proposes several competing strategies; the gate tests them in rank
order with fall-through. We script, per round, a *losing* first strategy and a
*winning* second one, so the test proves the loop: (a) diagnoses, (b) proposes
competitors, (c) discards the loser at the gate/structural check, and (d) keeps
the winner — ending at the known optimum.
"""

from studio.backends.mock import MockBackend
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.benchmark import toy_fixes
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator

SPLIT = TaskSplit(
    held_in=[f"{f}-{i}" for f in FAMILIES for i in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)],
    held_out=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
)

# Per round: [strategy_0 (loses), strategy_1 (wins)]. Under do-no-harm the loser
# must *regress* (a neutral edit would now be accepted), so s0 is always a
# regression or a structural break.
STRATEGIST_ACTIONS = [
    toy_fixes.regress_echo, toy_fixes.enable_upper,   # round 1: gate rejects s0 (regression) -> s1
    toy_fixes.break_boot, toy_fixes.fix_reverse,      # round 2: structural skips s0 -> s1
    toy_fixes.regress_echo, toy_fixes.fix_add_full,   # round 3: gate rejects s0 (regression) -> s1
]
ROUNDS = 3

DIAG = [{
    "pattern_id": "p1", "description": "several operations fail",
    "root_cause": "buggy or disabled ops", "failing_task_ids": ["reverse-0"],
    "blamed_part": "tool_code", "confidence": 0.7,
}]


def _backend():
    return MockBackend(
        json_responses={
            "diagnoser": [DIAG] * ROUNDS,
            "reviewer": [{"keep": [], "drop": []}] * ROUNDS,   # keep all
            "ranker": [{"order": []}] * ROUNDS,                # -> input order: s0 then s1
        },
        agent_actions={"strategist": list(STRATEGIST_ACTIONS)},
    )


def _run(tmp_path):
    config = Config(
        loop=LoopConfig(rounds=ROUNDS, wobble_runs=3, strategies_per_round=2),
        edits=EditConfig(allow_repair=False),
    )
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=_backend(),
        config=config,
        split=SPLIT,
        part_map=toy_part_map(),
    )
    return orch.run(), orch


def test_loop_reaches_optimum(tmp_path):
    result, _ = _run(tmp_path)
    assert result.baseline_final == 0.25
    assert result.final_score == 1.0
    assert result.accepted == ROUNDS  # every round, the winner was kept


def test_winner_chosen_via_fallthrough(tmp_path):
    result, orch = _run(tmp_path)
    # In every round the second strategy (s1) won after the first lost.
    for r in result.rounds:
        assert r.accepted and "s1 accepted" in r.note
    # The broken first strategy in round 2 was recorded to the avoid-list.
    assert any("SyntaxError" in a for a in orch.state.avoid_list)
    assert orch.state.health.reward_hack_incidents == 0
