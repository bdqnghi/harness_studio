"""M4 integration: reward-hack halt and cache/cost instrumentation in the loop."""

from pathlib import Path

from studio.backends.mock import MockBackend
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.benchmark import toy_fixes
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator

SPLIT = TaskSplit(
    judging=[f"{f}-{i}" for f in FAMILIES for i in (0, 1)],
    final_exam=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
    audit=[f"{f}-{i}" for f in FAMILIES for i in (4, 5)],
    practice=[f"{f}-{i}" for f in FAMILIES for i in (6, 7, 8, 9, 10, 11)],
)
DIAG = [{"pattern_id": "p", "description": "x", "root_cause": "x",
         "failing_task_ids": ["reverse-0"], "blamed_part": "tool_code", "confidence": 0.5}]


class HackBench(ToyBenchmark):
    """Returns impossible scores once a candidate plants a HACK marker."""

    def run(self, harness, task_ids, *, run_idx=0):
        if "HACK" in harness.read_file("tools.py"):
            return {t: 5.0 for t in task_ids}
        return super().run(harness, task_ids, run_idx=run_idx)


def inject_hack(root: Path) -> None:
    p = Path(root) / "tools.py"
    p.write_text(p.read_text() + "\n# HACK\n")


def _orch(tmp_path, bench, actions, **loop):
    backend = MockBackend(
        json_responses={"diagnoser": [DIAG] * 8, "reviewer": [{"keep": [], "drop": []}] * 8},
        agent_actions={"strategist": actions},
    )
    config = Config(loop=LoopConfig(wobble_runs=2, strategies_per_round=1, **loop),
                    edits=EditConfig(allow_repair=False))
    return Orchestrator(
        workspace=tmp_path / "ws", source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=bench, backend=backend, config=config, split=SPLIT,
        part_map=toy_part_map(),
    )


def test_reward_hack_halts_the_run(tmp_path):
    orch = _orch(tmp_path, HackBench(per_family=12), [inject_hack], rounds=3)
    result = orch.run()
    assert result.halted
    assert orch.state.health.reward_hack_incidents == 1
    assert any("HALT reward_hack" in line for line in orch.state.health_log)


def test_cost_and_cache_instrumented(tmp_path):
    orch = _orch(tmp_path, ToyBenchmark(per_family=12), [toy_fixes.fix_reverse], rounds=1)
    result = orch.run()
    assert result.task_runs > 0
    assert result.cache_hits > 0          # the stable judging set is re-scored from cache
    assert result.uplift > 0
    assert result.cost_per_point < float("inf")
