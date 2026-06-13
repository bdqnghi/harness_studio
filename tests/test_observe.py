"""Observability: a (tree) run emits a tail-able progress.jsonl and persists
health signals — the only live view into a multi-hour optimization."""

import json

from studio.benchmark.toy import ToyBenchmark
from tests.test_tree_loop import _run_tree

ROUNDS = 3  # _run_tree's default


def _events(orch):
    return [json.loads(line) for line in orch.state.progress_path.read_text().splitlines()]


def test_progress_jsonl_event_stream(tmp_path):
    _, orch = _run_tree(tmp_path)
    events = _events(orch)
    assert all("ts" in e and "event" in e for e in events)
    names = [e["event"] for e in events]
    assert names[0] == "run_start"
    assert "setup_done" in names
    # Every round produces the full sequence.
    for r in range(1, ROUNDS + 1):
        rnames = [e["event"] for e in events if e.get("round") == r]
        for expected in ("round_start", "batch_done", "diagnosis_done",
                         "proposal_done", "acceptance_decision", "round_end"):
            assert expected in rnames, f"round {r} missing {expected}: {rnames}"
    # The tree accepts at least one edit (rounds 2-3).
    assert any(e["event"] == "acceptance_decision" and e.get("accept") for e in events)
    # The trailing segment is audited and reported.
    assert any(e["event"] == "segment_boundary" for e in events)
    # round_end carries cumulative cost counters.
    ends = [e for e in events if e["event"] == "round_end"]
    assert all("task_runs" in e and "cache_hits" in e for e in ends)


def test_health_log_persists_to_disk(tmp_path):
    _, orch = _run_tree(tmp_path)
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
                return {t: 9.0 for t in scores}  # impossible -> reward-hack guard trips
            return scores

    result, orch = _run_tree(
        tmp_path, benchmark=HackBench(per_family=12, noise_per_mille=0), rounds=1,
    )
    assert result.halted
    events = _events(orch)
    assert events[-1]["event"] == "halt" and events[-1]["reason"] == "reward_hack"
    assert "HALT reward_hack" in orch.state.health_log_path.read_text()
