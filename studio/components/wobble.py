"""Wobble measurement (PRD §5.0b): establish the benchmark's noise floor.

Run the unchanged harness on a fixed set N times and take the spread of the
aggregate score. The gate uses this scalar to avoid mistaking noise for a real
gain. Re-calibrate occasionally (PRD §5.0b, §7).
"""

from __future__ import annotations

from ..benchmark.base import Benchmark
from ..harness import Harness


def measure_wobble(
    benchmark: Benchmark, harness: Harness, task_ids: list[str], runs: int = 5
) -> float:
    """Return the spread (max - min) of the aggregate score across ``runs``."""
    aggregates = []
    for i in range(max(1, runs)):
        scores = benchmark.run(harness, task_ids, run_idx=i)
        aggregates.append(sum(scores.values()) / len(scores) if scores else 0.0)
    return max(aggregates) - min(aggregates)
