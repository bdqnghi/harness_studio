"""Task splitter (PRD §5.0c, §6): partition tasks into four disjoint piles.

Keeping the piles disjoint is what stops the harness from "winning" by
overfitting the exact tasks it is repeatedly scored on. The final-exam pile is
carved first and never touched until the end (the one honest number).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import PileConfig


@dataclass
class TaskSplit:
    practice: list[str]  # pool sampled fresh each round (Runner)
    judging: list[str]  # stable within a segment (Gate)
    audit: list[str]  # large, mostly untouched (Deep auditor)
    final_exam: list[str]  # locked until the very end (Final report)


def _ordering(task_ids: list[str], seed: int) -> list[str]:
    """Deterministic, seed-dependent shuffle (stable across machines)."""

    def key(tid: str) -> str:
        return hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()

    return sorted(task_ids, key=key)


def split_tasks(task_ids: list[str], piles: PileConfig, seed: int = 0) -> TaskSplit:
    """Carve the piles in priority order: final_exam, audit, judging, practice."""
    order = _ordering(task_ids, seed)
    take = lambda n: [order.pop(0) for _ in range(min(n, len(order)))]  # noqa: E731
    final_exam = take(piles.final_exam)
    audit = take(piles.audit)
    judging = take(piles.judging)
    practice = list(order)  # everything left is the practice pool
    return TaskSplit(practice=practice, judging=judging, audit=audit, final_exam=final_exam)


def sample_practice(split: TaskSplit, size: int, seed: int, round_idx: int) -> list[str]:
    """Fresh-random practice batch for a round (deterministic given seed+round)."""
    order = _ordering(split.practice, seed * 1000 + round_idx)
    return order[:size]
