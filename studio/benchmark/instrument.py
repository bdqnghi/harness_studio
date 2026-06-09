"""Instrumented benchmark wrapper: caching, cost counters, reward-hack defense.

Task runs are the expensive resource (PRD §4, §8), so we (a) cache scores by
``(harness_hash, run_idx, task)`` — the judging set is stable within a segment, so
re-scoring the unchanged old harness each round hits the cache — and (b) count the
task runs actually executed, to report cost-per-point (PRD §9.2).

It is also the chokepoint for the reward-hacking defense (PRD §3, §7): any score
outside [0, 1] is impossible from an honest evaluator, so we raise immediately and
let the orchestrator halt.
"""

from __future__ import annotations

from ..harness import Harness
from .base import Benchmark


class RewardHackError(RuntimeError):
    """An impossible score was observed — the run halts (PRD §7)."""


class InstrumentedBenchmark(Benchmark):
    def __init__(self, inner: Benchmark, *, cache: bool = True) -> None:
        self.inner = inner
        self.cache_enabled = cache
        self._cache: dict[tuple[str, int, str], float] = {}
        self.task_runs = 0  # task-score evaluations actually executed
        self.cache_hits = 0

    def list_tasks(self) -> list[str]:
        return self.inner.list_tasks()

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        return self.inner.boot_check(harness)

    def describe(self, task_id: str) -> str:
        return self.inner.describe(task_id)

    def last_trace(self, task_id: str) -> str:
        return self.inner.last_trace(task_id)

    def run(self, harness, task_ids, *, run_idx=0):
        h = harness.content_hash()
        out: dict[str, float] = {}
        missing: list[str] = []
        for t in task_ids:
            key = (h, run_idx, t)
            if self.cache_enabled and key in self._cache:
                out[t] = self._cache[key]
                self.cache_hits += 1
            else:
                missing.append(t)

        if missing:
            fresh = self.inner.run(harness, missing, run_idx=run_idx)
            self.task_runs += len(missing)
            for t in missing:
                score = fresh[t]
                if not (0.0 <= score <= 1.0):
                    raise RewardHackError(
                        f"impossible score {score!r} for task {t!r}"
                    )
                self._cache[(h, run_idx, t)] = score
                out[t] = score
        return out
