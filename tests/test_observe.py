"""Observability: a run emits a tail-able progress.jsonl and persists health
signals — the only live view into a multi-hour optimization."""

import json

from studio.backends.mock import MockBackend
from studio.benchmark import toy_fixes
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator

SPLIT = TaskSplit(
    held_in=[f"{f}-{i}" for f in FAMILIES for i in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)],
    held_out=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
)

DIAG = [{
    "pattern_id": "p1", "description": "several operations fail",
    "root_cause": "buggy or disabled ops", "failing_task_ids": ["reverse-0"],
    "blamed_part": "tool_code", "confidence": 0.7,
}]

ROUNDS = 2


def _run(tmp_path):
    backend = MockBackend(
        json_responses={
            "diagnoser": [DIAG] * ROUNDS,
            "reviewer": [{"keep": [], "drop": []}] * ROUNDS,
            "ranker": [{"order": []}] * ROUNDS,
        },
        agent_actions={"strategist": [
            toy_fixes.regress_echo, toy_fixes.enable_upper,
            toy_fixes.regress_echo, toy_fixes.fix_reverse,
        ]},
    )
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=backend,
        config=Config(
            loop=LoopConfig(rounds=ROUNDS, wobble_runs=3, strategies_per_round=2),
            edits=EditConfig(allow_repair=False),
        ),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    result = orch.run()
    return result, orch


def _events(orch):
    lines = orch.state.progress_path.read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_progress_jsonl_event_stream(tmp_path):
    _, orch = _run(tmp_path)
    events = _events(orch)
    assert all("ts" in e and "event" in e for e in events)
    names = [e["event"] for e in events]
    assert names[0] == "run_start"
    assert "setup_done" in names
    # Every round produces the full sequence.
    for r in range(1, ROUNDS + 1):
        rnd = [e for e in events if e.get("round") == r]
        rnames = [e["event"] for e in rnd]
        for expected in ("round_start", "batch_done", "diagnosis_done",
                         "proposal_done", "gate_decision", "round_end"):
            assert expected in rnames, f"round {r} missing {expected}: {rnames}"
    # The losing strategy then the winner -> at least 2 gate decisions in round 1.
    r1_gates = [e for e in events if e["event"] == "gate_decision" and e["round"] == 1]
    assert len(r1_gates) == 2
    assert r1_gates[0]["accept"] is False and r1_gates[1]["accept"] is True
    # The trailing segment is audited and reported.
    assert any(e["event"] == "segment_boundary" for e in events)
    # round_end carries cumulative cost counters.
    ends = [e for e in events if e["event"] == "round_end"]
    assert all("task_runs" in e and "cache_hits" in e for e in ends)


def test_health_log_persists_to_disk(tmp_path):
    _, orch = _run(tmp_path)
    orch.state.log_health("demo_signal: something happened -> investigate")
    text = orch.state.health_log_path.read_text()
    assert "demo_signal" in text
    assert "demo_signal: something happened -> investigate" in orch.state.health_log


def test_reward_hack_halt_is_last_event(tmp_path):
    class HackBench(ToyBenchmark):
        """Honest on the baseline harness; impossible scores for any candidate,
        so the halt fires inside the round loop (where it is catchable)."""

        _baseline_hash = None

        def run(self, harness, task_ids, *, run_idx=0):
            h = harness.content_hash()
            if self._baseline_hash is None:
                self._baseline_hash = h
            scores = super().run(harness, task_ids, run_idx=run_idx)
            if h != self._baseline_hash:
                return {t: 9.0 for t in scores}  # impossible scores
            return scores

    backend = MockBackend(
        json_responses={"diagnoser": [DIAG], "reviewer": [{"keep": [], "drop": []}]},
        agent_actions={"strategist": [toy_fixes.enable_upper]},
    )
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=HackBench(per_family=12, noise_per_mille=0),
        backend=backend,
        config=Config(loop=LoopConfig(rounds=1, wobble_runs=2, strategies_per_round=1)),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    result = orch.run()
    assert result.halted
    events = _events(orch)
    assert events[-1]["event"] == "halt" and events[-1]["reason"] == "reward_hack"
    assert "HALT reward_hack" in orch.state.health_log_path.read_text()
