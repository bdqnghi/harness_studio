"""Task splitter: carve a benchmark into the three working sets.

The optimizer has a **fixed appetite** (like an SGD mini-batch): the set it sees
every round does NOT grow with the benchmark size. So a benchmark is split into
three sets only:

  * **held_in**    — the pool you optimize on. Each round samples ``round_size``
    tasks from it to find failures; the gate scores old-vs-new on it.
  * **regression** — a disjoint do-no-harm set, pooled with held_in into the
    gate's NET decision (an edit must not make the whole pool worse).
  * **held_out**   — locked the whole time, graded once at the end. The only
    honest number.

There is deliberately **no separate "judging"/"audit" set**: the gate just scores
on held_in, and the noise that a held-aside audit slice would (weakly) guard
against is handled the right way — by *re-measurement* (repeated rollouts on
borderline calls; a segment-boundary re-roll of the live harness) — since the
dominant variance here is measurement noise, not task-sampling. This matches the
prior art (Self-Harness, Arbor both use exactly held-in + held-out).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from ..config import PileConfig


@dataclass
class TaskSplit:
    held_in: list[str]  # the pool: sampled each round; the gate scores old-vs-new here
    regression: list[str] = field(default_factory=list)  # disjoint do-no-harm, pooled into the gate
    held_out: list[str] = field(default_factory=list)  # locked until the very end (the one honest number)


def _ordering(task_ids: list[str], seed: int) -> list[str]:
    """Deterministic, seed-dependent shuffle (stable across machines)."""

    def key(tid: str) -> str:
        return hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()

    return sorted(task_ids, key=key)


def split_tasks(task_ids: list[str], piles: PileConfig, seed: int = 0) -> TaskSplit:
    """Fixed-size fallback split (used when no explicit split is supplied).

    Carves in priority order: held_out, regression, held_in (everything left)."""
    order = _ordering(task_ids, seed)
    take = lambda n: [order.pop(0) for _ in range(min(n, len(order)))]  # noqa: E731
    held_out = take(piles.held_out)
    regression = take(piles.regression)
    held_in = list(order)  # everything left is the held-in pool
    return TaskSplit(held_in=held_in, regression=regression, held_out=held_out)


def sample_held_in(split: TaskSplit, size: int, seed: int, round_idx: int) -> list[str]:
    """Fresh-random held-in batch for a round (deterministic given seed+round)."""
    order = _ordering(split.held_in, seed * 1000 + round_idx)
    return order[:size]


def detectable_delta(n: int, sigma2: float, *, z: float = 1.96, k: int = 3) -> float:
    """Smallest paired effect reliably resolvable with ``n`` tasks at ``k`` rollouts
    and confidence ``z``: sqrt(z^2 * 2*sigma2 / (k * n)). Used to report the
    detectable floor for the per-round gate and the locked-test verdict."""
    return math.sqrt((z * z * 2.0 * sigma2) / (max(1, k) * max(1, n)))
