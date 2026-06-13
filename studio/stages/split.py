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

from studio.config import PileConfig


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


def random_split(tasks: list[str], *, seed: int, held_in: int, reg: int,
                 held_out_cap: int = 0) -> TaskSplit:
    """Blind (no-profile) 3-set split: held_in is a small fixed scoop, regression
    a disjoint slice, held_out the (capped) locked rest. Used with --no-profile."""
    order = _ordering(tasks, seed)
    hi = min(held_in, max(0, len(order) - reg - 4))  # leave >=4 for a locked test
    held_in_set = order[:hi]
    regression = order[hi:hi + reg]
    rest = order[hi + reg:]
    held_out = rest[:held_out_cap] if held_out_cap else rest
    if not held_in_set:  # degenerate tiny set: fall back to using regression
        held_in_set, regression = regression, []
    return TaskSplit(held_in=held_in_set, regression=regression, held_out=held_out)


def _stratified_take(bins: dict[str, list[str]], n: int) -> list[str]:
    """Take ``n`` items proportionally across bins (largest-remainder), preserving
    each bin's (already-seeded) order. Representative sample across difficulty."""
    total = sum(len(v) for v in bins.values())
    if n <= 0 or total == 0:
        return []
    n = min(n, total)
    raw = {k: n * len(v) / total for k, v in bins.items()}
    base = {k: int(raw[k]) for k in bins}
    rem = n - sum(base.values())
    for k in sorted(bins, key=lambda k: (raw[k] - base[k], k), reverse=True)[:rem]:
        base[k] += 1
    out: list[str] = []
    for k, v in bins.items():
        out.extend(v[:base[k]])
    return out


def stratified_split(profile, *, held_in: int, reg: int, held_out_cap: int = 0,
                     seed: int = 0, solved: float = 0.8, failing: float = 0.2) -> TaskSplit:
    """Difficulty-stratified 3-set split from a :class:`profiler.Profile`.

    - **held_out**: a representative (proportional) sample across solved/mixed/
      failing — locked first so the final number is unbiased.
    - **regression**: reliably-SOLVED tasks not in held_out (do-no-harm guard).
    - **held_in**: FAILING then MIXED then (top-up) solved, not already taken —
      so the optimizer always has learnable failures to work on.

    Deterministic given ``seed``; the three sets are disjoint."""
    bins = {"solved": [], "mixed": [], "failing": []}
    for t in profile.pass_rate:
        bins[profile.bin(t, solved=solved, failing=failing)].append(t)
    bins = {k: _ordering(v, seed) for k, v in bins.items()}
    all_tasks = _ordering(list(profile.pass_rate), seed)
    n = len(all_tasks)

    ho_n = held_out_cap or max(0, n - held_in - reg)
    held_out = _stratified_take(bins, ho_n)
    taken = set(held_out)

    regression = [t for t in bins["solved"] if t not in taken][:reg]
    taken |= set(regression)

    learnable = ([t for t in bins["failing"] if t not in taken]
                 + [t for t in bins["mixed"] if t not in taken]
                 + [t for t in bins["solved"] if t not in taken])  # top-up if too few failures
    held_in_set = learnable[:held_in]
    if not held_in_set:  # degenerate: borrow back from whatever's left
        held_in_set = [t for t in all_tasks if t not in taken][:max(1, held_in)]
    return TaskSplit(held_in=held_in_set, regression=regression, held_out=held_out)
