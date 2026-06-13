"""Runner (PRD §5.1): gather fresh failures for a round.

Runs the current harness once over the round's held-in batch and reports which
failed. These scores *locate failures only* — they make no keep/reject decision
(that is the gate's job, with precision reserved for it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmark.base import Benchmark
from ..harness import Harness


@dataclass
class Failure:
    task_id: str
    description: str
    trace: str = ""  # excerpt of why it failed (verifier output + agent trajectory)


@dataclass
class RunReport:
    scores: dict[str, float]
    failures: list[Failure] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def run_batch(
    benchmark: Benchmark, harness: Harness, task_ids: list[str]
) -> RunReport:
    scores = benchmark.run(harness, task_ids, run_idx=0)
    failures = [
        Failure(tid, benchmark.describe(tid), trace=benchmark.last_trace(tid, harness=harness))
        for tid, s in scores.items()
        if s < 1.0
    ]
    return RunReport(scores=scores, failures=failures)
