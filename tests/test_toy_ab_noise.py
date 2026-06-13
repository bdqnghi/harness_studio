"""Toy A/B (experiment phase T3): both arms under injected noise and noiseless.

Equal-quality scripted proposers feed both arms the same edit sequence. Under
noise (noise_per_mille=120 ≈ σ 0.11 flip noise), across 5 seeds, neither arm
may ever accept the known regression (disabling echo). Noiseless, both arms
must reach the toy optimum, with the tree spending no more Tier-A runs than
classic.
"""

import pytest

from studio.backends.mock import MockBackend
from studio.benchmark import toy_fixes
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator

SPLIT = TaskSplit(
    judging=[f"{f}-{i}" for f in FAMILIES for i in (0, 1)],
    final_exam=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
    audit=[f"{f}-{i}" for f in FAMILIES for i in (4, 5)],
    practice=[f"{f}-{i}" for f in FAMILIES for i in (6, 7, 8, 9, 10, 11)],
)

DIAG = [{
    "pattern_id": "p1", "description": "ops fail", "root_cause": "buggy or disabled ops",
    "failing_task_ids": ["reverse-6"], "blamed_part": "tool_code", "confidence": 0.8,
    "verifier_cause": "wrong output", "agent_mechanism": "op missing or buggy",
    "addressable": True,
}]

# The same edit sequence for both arms: three known-good fixes, then the
# known regression that the gate must reject under any noise.
ACTIONS = [toy_fixes.enable_upper, toy_fixes.fix_reverse, toy_fixes.fix_add_full,
           toy_fixes.regress_echo]
ROUNDS = len(ACTIONS)
SURPLUS = ROUNDS * 3  # scripted Tier-B responses beyond the worst-case need


def _classic_backend():
    return MockBackend(
        json_responses={
            "diagnoser": [DIAG] * SURPLUS,
            "reviewer": [{"keep": [], "drop": []}] * SURPLUS,
        },
        agent_actions={"strategist": list(ACTIONS)},
    )


def _tree_backend():
    hyp = {"hypotheses": [{
        "title": "next fix", "mechanism": "m",
        "hypothesis": "apply the next scripted toy fix", "observable": "scores move",
    }]}
    return MockBackend(
        json_responses={
            "diagnoser": [DIAG] * SURPLUS,
            "direction-router": (
                [{"assignments": [{"pattern_id": "p1", "direction_id": "",
                                   "new_title": "broken ops", "new_mechanism": "m"}]}]
                + [{"assignments": [{"pattern_id": "p1", "direction_id": "d1"}]}] * SURPLUS
            ),
            "ideator": [hyp] * SURPLUS,
            "insight": [{"insight": "lesson"}] * SURPLUS,
            "insight-direction": [{"insight": "summary"}] * SURPLUS,
        },
        agent_actions={"strategist": list(ACTIONS)},
    )


def _run(tmp_path, *, optimizer, seed, noise, name):
    backend = _tree_backend() if optimizer == "tree" else _classic_backend()
    orch = Orchestrator(
        workspace=tmp_path / f"ws_{name}",
        source_harness=build_toy_harness(tmp_path / f"src_{name}"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=noise),
        backend=backend,
        config=Config(
            seed=seed,
            loop=LoopConfig(rounds=ROUNDS, wobble_runs=3, strategies_per_round=1,
                            optimizer=optimizer, hypotheses_per_direction=1),
            edits=EditConfig(allow_repair=False),
        ),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    return orch.run(), orch


@pytest.mark.parametrize("seed", range(5))
def test_noisy_ab_never_accepts_the_known_regression(tmp_path, seed):
    for optimizer in ("classic", "tree"):
        result, orch = _run(tmp_path, optimizer=optimizer, seed=seed,
                            noise=120, name=f"{optimizer}{seed}")
        instructions = orch.harness.read_file("instructions.txt")
        assert "ENABLE echo" in instructions, (
            f"{optimizer} seed {seed}: the known regression was accepted"
        )
        # The regression round (the 4th scripted edit) was never an accept.
        regress_rounds = [r for r in result.rounds if r.round_idx == ROUNDS]
        assert all(not r.accepted for r in regress_rounds)


def test_noiseless_both_arms_reach_optimum_tree_spends_less_tier_a(tmp_path):
    res_c, orch_c = _run(tmp_path, optimizer="classic", seed=0, noise=0, name="c")
    res_t, orch_t = _run(tmp_path, optimizer="tree", seed=0, noise=0, name="t")
    assert res_c.final_score == 1.0 and res_t.final_score == 1.0
    assert res_c.accepted == 3 and res_t.accepted == 3  # the 3 good fixes
    tier_a = lambda orch: sum(1 for k, t in orch.backend.calls
                              if k == "run_agent" and t == "strategist")
    assert tier_a(orch_t) <= tier_a(orch_c)
