"""The benchmark seam — how a harness gets scored.

The acceptance (PRD §5.8) decides keep/reject purely from numbers this interface
returns, so the benchmark is the trust anchor: deterministic code, no AI. Real
targets (Terminus-KIRA on Terminal-Bench) and the toy target both implement it,
so the loop never knows which it is running against.
"""

from __future__ import annotations

import abc

from ..core.harness import Harness


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
        noise the acceptance must see through)."""

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        """Cheap structural check: does the harness compile/load/boot?

        Returns (ok, error_message). Runs no tasks → free. Default: always ok;
        targets override (toy execs its tools; Kira imports its module)."""
        return True, ""

    def describe(self, task_id: str) -> str:
        """Human-readable description of a task, shown to the Diagnoser/Strategist.

        Default is the id; targets enrich it (the toy adds input -> expected)."""
        return task_id

    def last_trace(self, task_id: str, *, harness: Harness | None = None) -> str:
        """A concise excerpt of why ``task_id`` failed on its most recent run —
        the verifier output and the agent's last actions. Fed to the Diagnoser so
        it blames a real cause, not just the task name. Default: none.

        ``harness`` scopes the lookup: traces are versioned per harness so a
        candidate's acceptance run can never be attributed to the live harness.

        Real targets override this; it must degrade gracefully (return "") when no
        trace is available, so the loop never depends on it."""
        return ""

    # Structured failure evidence (core/evidence.py). Adapters that decode
    # their verifier output set ``evidence_store`` and override ``last_evidence``;
    # everything else falls back to the flat ``last_trace`` above. The localizer
    # and editor consume this — see stages/optimize/localizer.py.
    evidence_store = None  # type: ignore[assignment]

    def last_evidence(self, task_id: str, *, harness: Harness | None = None):
        """Structured :class:`~studio.core.evidence.TaskEvidence` for why
        ``task_id`` failed, or ``None`` when unavailable. Default: none."""
        return None
