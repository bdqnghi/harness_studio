"""The benchmark seam — how a harness gets scored.

The gate (PRD §5.8) decides keep/reject purely from numbers this interface
returns, so the benchmark is the trust anchor: deterministic code, no AI. Real
targets (Terminus-KIRA on Terminal-Bench) and the toy target both implement it,
so the loop never knows which it is running against.
"""

from __future__ import annotations

import abc

from ..harness import Harness


class Benchmark(abc.ABC):
    """Scores a harness on tasks. Scores are in [0, 1]; higher is better."""

    @abc.abstractmethod
    def list_tasks(self) -> list[str]:
        """All task ids available, in a stable order."""

    @abc.abstractmethod
    def run(
        self, harness: Harness, task_ids: list[str], *, run_idx: int = 0
    ) -> dict[str, float]:
        """Run ``harness`` on the given tasks once and return per-task scores.

        ``run_idx`` distinguishes repeated runs of the same harness/tasks; a
        flaky benchmark may return different scores per run_idx (this is the
        noise the gate must see through)."""

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        """Cheap structural check: does the harness compile/load/boot?

        Returns (ok, error_message). Runs no tasks → free. Default: always ok;
        targets override (toy execs its tools; Kira imports its module)."""
        return True, ""

    def describe(self, task_id: str) -> str:
        """Human-readable description of a task, shown to the Diagnoser/Strategist.

        Default is the id; targets enrich it (the toy adds input -> expected)."""
        return task_id
