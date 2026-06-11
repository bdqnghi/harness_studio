"""Task splitter: carve a benchmark into the three working sets.

The optimizer has a **fixed appetite** (like an SGD mini-batch): the sets it
sees every round do NOT grow with the benchmark size. So a benchmark is split
into three sets only:

  * **held-in pool** (``practice``)  — the variety pool; each round samples
    ``round_size`` tasks from it to find failures AND first-check an edit.
  * **regression**                   — a disjoint do-no-harm set; an edit must
    not hurt it (the second, independent gate check).
  * **held-out test** (``final_exam``) — locked the whole time, graded once at
    the end. This is the only honest number.

``judging`` is a stable, power-sized slice of the held-in pool (the gate scores
old-vs-new on it every round); ``audit`` is a small slice of the pool the deep
auditor periodically re-checks. Slow/"heavy" tasks go ONLY into the locked test
(never the every-round sets), so a 3-hour task can be graded but can never stall
a round.

Held-in (pool + regression) is a roughly constant scoop (≤ a few hundred) no
matter how big N is; everything else goes to the locked test, so **more data
buys a sharper final number, not a slower optimizer.** Below a floor where an
honest held-out can't be seated, the plan switches to *transfer* mode (optimize
on all, verify on a different benchmark/model).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from ..config import PileConfig


@dataclass
class TaskSplit:
    practice: list[str]  # held-in pool sampled fresh each round (Runner)
    judging: list[str]  # stable gate set (primary do-no-harm check) ⊆ practice
    audit: list[str]  # small pool slice the deep auditor re-checks ⊆ practice
    final_exam: list[str]  # locked until the very end (the one honest number)
    regression: list[str] = field(default_factory=list)  # disjoint do-no-harm 2nd check


def _ordering(task_ids: list[str], seed: int) -> list[str]:
    """Deterministic, seed-dependent shuffle (stable across machines)."""

    def key(tid: str) -> str:
        return hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()

    return sorted(task_ids, key=key)


def split_tasks(task_ids: list[str], piles: PileConfig, seed: int = 0) -> TaskSplit:
    """Fixed-size fallback split (used when no adaptive plan is supplied).

    Carves in priority order: final_exam, audit, judging, practice."""
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


# === power-based, calibration-aware planning ===================================
#
# Held-in size comes from STATISTICAL POWER + an affordability cap, so it is
# ~constant across N (an SGD mini-batch), not a fraction of N. Surplus tasks in a
# big benchmark go to the locked test (scored once), buying test precision.


@dataclass
class SplitPlan:
    """A size-aware evaluation plan chosen from the benchmark itself.

    ``mode == "holdout"``: a single fixed split with a locked test (use ``.split``).
    ``mode == "transfer"``: N too small for an honest held-out — optimize on all,
    verify generalization on a different benchmark/model (caller's job).
    """

    mode: str
    k: int                          # rollouts for the final graded test (test_k)
    split: TaskSplit | None = None
    rationale: str = ""
    sigma2: float = 0.0
    n_pool: int = 0                 # held-in variety pool (~constant across N)
    n_judging: int = 0              # stable gate set ⊆ pool
    n_regression: int = 0           # disjoint do-no-harm set
    n_test: int = 0                 # locked, graded once
    detectable_round: float = 0.0   # smallest effect the per-round gate can resolve
    detectable_final: float = 0.0   # smallest effect the locked-test verdict can resolve
    recommend: str = ""             # "split" | "transfer"


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def power_n(sigma2: float, *, z: float = 1.96, delta: float = 0.1, k: int = 3) -> int:
    """Tasks (at k rollouts) needed to resolve a paired effect ``delta`` at noise
    ``sigma2`` and confidence ``z``: n = z^2 * 2*sigma2 / (k * delta^2)."""
    if delta <= 0:
        return 10 ** 9
    return math.ceil((z * z * 2.0 * sigma2) / (max(1, k) * delta * delta))


def detectable_delta(n: int, sigma2: float, *, z: float = 1.96, k: int = 3) -> float:
    """Smallest effect reliably resolvable with ``n`` tasks at ``k`` rollouts."""
    return math.sqrt((z * z * 2.0 * sigma2) / (max(1, k) * max(1, n)))


def _strata(task_ids: list[str], difficulties: dict[str, float] | None, n_bins: int = 3) -> dict[str, int]:
    """Bin tasks into ``n_bins`` equal-count difficulty strata (by pass-rate p).
    Tasks with no difficulty go to a single stratum (unstratified)."""
    if not difficulties:
        return {t: 0 for t in task_ids}
    ranked = sorted(task_ids, key=lambda t: (difficulties.get(t, 0.5), t))
    n = len(ranked)
    return {t: min(n_bins - 1, i * n_bins // max(1, n)) for i, t in enumerate(ranked)}


def _stratified_sample(pool: list[str], n: int, strata: dict[str, int], seed: int) -> list[str]:
    """Deterministically take ``n`` tasks from ``pool``, proportional across strata
    (largest-remainder), each stratum ordered by the seeded shuffle. Representative."""
    n = min(max(0, n), len(pool))
    if n == 0:
        return []
    groups: dict[int, list[str]] = {}
    for t in pool:
        groups.setdefault(strata.get(t, 0), []).append(t)
    groups = {s: _ordering(g, seed) for s, g in groups.items()}
    total = len(pool)
    raw = {s: n * len(g) / total for s, g in groups.items()}
    base = {s: int(raw[s]) for s in groups}
    rem = n - sum(base.values())
    for s in sorted(groups, key=lambda s: (raw[s] - base[s], s), reverse=True)[:rem]:
        base[s] += 1
    picked: list[str] = []
    for s, g in groups.items():
        picked.extend(g[:base[s]])
    return picked


def choose_split(
    task_ids: list[str],
    *,
    sigma2: float,
    round_size: int = 32,
    difficulties: dict[str, float] | None = None,
    timeouts: dict[str, float] | None = None,
    seed: int = 0,
    opt_k: int = 1,
    test_k: int = 3,
    z: float = 1.96,
    delta_round: float = 0.12,
    val_floor: int = 8,
    reg_floor: int = 16,
    reg_cap: int = 32,
    pool_mult: int = 4,
    pool_cap: int = 256,
    test_floor: int = 25,
    test_budget_cap: int = 0,
    heavy_sec: float = 3600.0,
) -> SplitPlan:
    """Carve the benchmark into held-in pool + regression + locked test.

    The held-in scoop (``pool`` + ``regression``) is sized by power and capped, so
    it stays ~constant as N grows; the locked test absorbs everything else. Slow
    tasks (``timeout >= heavy_sec``) go ONLY to the test. Below a floor where an
    honest held-out can't be seated, returns ``mode="transfer"``.

    Sizing knobs:
      * ``round_size`` — tasks run per round (the SGD mini-batch). Default 32.
      * ``pool`` = clamp(``pool_mult*round_size``, round_size, ``pool_cap``).
      * ``regression`` = clamp(power_n(δ=``delta_round``), ``reg_floor``, ``reg_cap``).
      * ``judging`` (stable gate slice ⊆ pool) = clamp(power_n, ``val_floor``, round_size).
      * ``test`` = all leftover (incl. every heavy task); a ``test_budget_cap`` (>0)
        keeps every heavy task and grades a representative light-task subsample.
    """
    task_ids = list(dict.fromkeys(task_ids))
    N = len(task_ids)
    if round_size <= 0:
        raise ValueError("round_size must be positive")
    if opt_k <= 0 or test_k <= 0:
        raise ValueError("opt_k and test_k must be positive")
    if reg_floor < 0 or reg_cap < reg_floor:
        raise ValueError("reg_cap must be >= reg_floor >= 0")
    if pool_mult <= 0 or pool_cap < round_size:
        raise ValueError("pool_mult must be positive and pool_cap must be >= round_size")
    if test_floor < 0 or test_budget_cap < 0:
        raise ValueError("test_floor and test_budget_cap must be non-negative")
    sigma2 = max(0.01, min(0.25, sigma2))
    timeouts = timeouts or {}
    strata = _strata(task_ids, difficulties)
    order = _ordering(task_ids, seed)

    heavy = [t for t in order if timeouts.get(t, 0.0) >= heavy_sec]  # always test-only
    light = [t for t in order if timeouts.get(t, 0.0) < heavy_sec]
    L = len(light)

    # --- fixed scoops (do NOT grow with N) ---
    reg_n = _clamp(power_n(sigma2, z=z, delta=delta_round, k=opt_k), reg_floor, reg_cap)
    pool_n = _clamp(pool_mult * round_size, round_size, pool_cap)
    n_judging = _clamp(power_n(sigma2, z=z, delta=delta_round, k=opt_k), val_floor, round_size)

    # The locked test must reach test_floor; the heavy tasks (all go to test) count.
    test_light_floor = max(0, test_floor - len(heavy))

    # --- shrink-to-fit for small benchmarks ---
    # Light-task priority: regression (independent) -> pool (held-in) -> test.
    deficit = (reg_n + pool_n + test_light_floor) - L
    if deficit > 0:
        cut = min(deficit, pool_n - round_size)   # shrink pool first, keep >= round_size
        pool_n -= cut
        deficit -= cut
    if deficit > 0:
        cut = min(deficit, reg_n - reg_floor)      # then regression, keep >= reg_floor
        reg_n -= cut
        deficit -= cut

    if deficit > 0 or L < (round_size + reg_floor):
        # Too few tasks to lock an honest held-out -> optimize on all, verify by
        # TRANSFER (a different benchmark/model — the caller's job).
        reg = _stratified_sample(light, min(reg_floor, max(0, L // 4)), strata, seed + 2)
        pool = [t for t in light if t not in set(reg)]
        judging = _stratified_sample(pool, min(len(pool), n_judging), strata, seed + 3)
        audit = _stratified_sample(pool, min(len(pool), max(4, n_judging // 2)), strata, seed + 4)
        split = TaskSplit(practice=pool, judging=judging, audit=audit,
                          final_exam=list(heavy), regression=reg)
        det_round = detectable_delta(len(judging), sigma2, z=z, k=opt_k)
        return SplitPlan(
            mode="transfer", k=test_k, split=split, sigma2=sigma2,
            n_pool=len(pool), n_judging=len(judging), n_regression=len(reg),
            n_test=len(split.final_exam), detectable_round=det_round,
            detectable_final=0.0, recommend="transfer",
            rationale=(f"N={N}: too small for an honest held-out ({L} light < "
                       f"{round_size}+{reg_floor}); optimize on all and verify by "
                       f"TRANSFER to another benchmark/model."),
        )

    # --- the normal single split ---
    reg = _stratified_sample(light, reg_n, strata, seed + 2)
    rest = [t for t in light if t not in set(reg)]
    pool = _stratified_sample(rest, pool_n, strata, seed + 1)
    taken = set(reg) | set(pool)
    test_light = [t for t in light if t not in taken]
    test_full = list(heavy) + test_light          # ALL heavy + leftover light
    if test_budget_cap and len(test_full) > test_budget_cap:
        required = max(test_floor, len(heavy))
        if test_budget_cap < required:
            raise ValueError(
                "test_budget_cap cannot satisfy the locked-test invariants: "
                f"need at least {required} slots for test_floor={test_floor} "
                f"and {len(heavy)} heavy tasks"
            )
        light_budget = test_budget_cap - len(heavy)
        test = list(heavy) + _stratified_sample(
            test_light, light_budget, strata, seed + 9
        )
    else:
        test = test_full
    judging = _stratified_sample(pool, min(len(pool), n_judging), strata, seed + 3)
    audit = _stratified_sample(pool, min(len(pool), max(4, n_judging // 2)), strata, seed + 4)
    split = TaskSplit(practice=pool, judging=judging, audit=audit,
                      final_exam=test, regression=reg)
    det_round = detectable_delta(len(judging), sigma2, z=z, k=opt_k)
    det_final = detectable_delta(len(test), sigma2, z=z, k=test_k)
    graded_light = len(test) - len(heavy)
    return SplitPlan(
        mode="holdout", k=test_k, split=split, sigma2=sigma2,
        n_pool=len(pool), n_judging=len(judging), n_regression=len(reg), n_test=len(test),
        detectable_round=det_round, detectable_final=det_final, recommend="split",
        rationale=(f"N={N}: held-in pool={len(pool)} (sample {round_size}/round) + "
                   f"regression={len(reg)} (do-no-harm) | test={len(test)} locked "
                   f"({len(heavy)} heavy + {graded_light} light, graded once at k={test_k}). "
                   f"judging={len(judging)} (stable gate). detectable: round~{det_round:.3f}, "
                   f"test~{det_final:.3f}."),
    )
