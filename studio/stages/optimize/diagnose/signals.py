"""FailureSignal: the verifiable, per-task analysis of one failing trajectory.

Built deterministically from a benchmark's structured ``TaskEvidence`` (the
verifier's own checks), so the *facts* in a FailureSignal are ground truth, not
an LLM guess. The diagnoser groups these by their failed-check ``signature`` and
only asks the LLM to *name* the groups — counts and membership stay deterministic.

Research basis: keep attribution grounded in verifier checks (LLM-judged
attribution is unreliable — Who&When/TRAIL); separate verifiable facts from
interpretation (Errudite); distinguish a consistent failure from flakiness so the
optimizer doesn't chase noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from studio.core.evidence import TaskEvidence, VerifierSignal, to_flat_excerpt

# A task scoring at/above this over its rollouts is treated as "really failing"
# (not flaky); between this and a pass it's mixed/variance, not a fixable mode.
FAILING_AT_OR_BELOW = 0.2


@dataclass
class FailureSignal:
    """Structured analysis of one failing task — the diagnosis spine's unit."""

    task_id: str
    reward: float                              # the (worst-trial) verifier reward
    failed: list[VerifierSignal] = field(default_factory=list)   # checks that failed (ground truth)
    gt_diff: list[str] = field(default_factory=list)             # "expected vs actual" lines
    failed_channels: list[str] = field(default_factory=list)     # distinct failed kinds, e.g. ["db"]
    window: str = ""                           # compact grounded trace excerpt
    consistent: bool = True                    # False => flaky/mixed (don't treat as a pattern)


def _diff_line(s: VerifierSignal) -> str:
    head = f"{s.kind}:{s.name}".rstrip(":") if s.name else s.kind
    return f"{head} — {s.detail}".strip(" —") if s.detail else head


def signature(fs: FailureSignal) -> frozenset:
    """The (kind, name) set of failed checks — tasks sharing it are the same mode.
    Empty failed set degrades to a single ('other','') so grouping still works."""
    if not fs.failed:
        return frozenset({("other", "")})
    return frozenset((s.kind, s.name) for s in fs.failed)


def from_evidence(ev: TaskEvidence, *, score: float | None = None) -> FailureSignal:
    """Deterministically build a FailureSignal from structured evidence.

    ``score`` is the task's mean reward over the round's rollouts (used to flag
    flakiness); when omitted we treat the task as a consistent failure."""
    failed = [s for s in ev.signals if not s.passed]
    consistent = score is None or score <= FAILING_AT_OR_BELOW
    return FailureSignal(
        task_id=ev.task_id,
        reward=ev.reward,
        failed=failed,
        gt_diff=[_diff_line(s) for s in failed],
        failed_channels=sorted({s.kind for s in failed}),
        window=to_flat_excerpt(ev),
        consistent=consistent,
    )
