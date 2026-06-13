"""Deep auditor (PRD §5.11): catch what the fast acceptance cannot.

At a segment boundary, score the current harness on the big audit set (large,
mostly-untouched, one run each — stability from breadth, not repeats). A harness
that quietly got worse there (lucky judging-set wins, or changes that fight each
other) is flagged so the orchestrator can rewind and mark the segment's accepted
families as traps. Genuinely-better harnesses become the new best.
"""

from __future__ import annotations

from dataclasses import dataclass

from studio.benchmark.base import Benchmark
from studio.core.harness import Harness


@dataclass
class AuditVerdict:
    score: float
    verdict: str  # "better" | "worse" | "same" (relative to the best so far)


def audit(
    benchmark: Benchmark,
    harness: Harness,
    audit_tasks: list[str],
    *,
    best_score: float | None,
    noise_floor: float,
) -> AuditVerdict:
    scores = benchmark.run(harness, audit_tasks, run_idx=0)
    score = sum(scores.values()) / len(scores) if scores else 0.0
    if best_score is None or score > best_score + noise_floor:
        return AuditVerdict(score, "better")
    if score < best_score - noise_floor:
        return AuditVerdict(score, "worse")
    return AuditVerdict(score, "same")
