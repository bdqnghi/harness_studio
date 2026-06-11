"""Task splitter (PRD §5.0c, §6): partition tasks into four disjoint piles.

Keeping the piles disjoint is what stops the harness from "winning" by
overfitting the exact tasks it is repeatedly scored on. The final-exam pile is
carved first and never touched until the end (the one honest number).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from ..config import PileConfig


@dataclass
class TaskSplit:
    practice: list[str]  # pool sampled fresh each round (Runner)
    judging: list[str]  # stable within a segment (Gate — split 1)
    audit: list[str]  # large, mostly untouched (Deep auditor)
    final_exam: list[str]  # locked until the very end (Final report)
    gen: list[str] = field(default_factory=list)  # Gate split 2 (dual-split generalization check)


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


# --- dynamic, benchmark-size-aware splitting -----------------------------------

@dataclass
class SplitPlan:
    """A size-aware evaluation plan chosen from the benchmark itself.

    ``mode == "holdout"``: one fixed disjoint split (use ``.split``).
    ``mode == "kfold"``: rotate the test slice across ``.folds`` (use each
    fold's ``final_exam`` as that fold's test set, the rest to optimize on).
    """

    mode: str
    k: int
    split: TaskSplit | None = None
    folds: list[TaskSplit] | None = None
    rationale: str = ""
    # power-based reporting (filled by choose_eval_plan)
    n_val: int = 0
    sigma2: float = 0.0
    detectable_step: float = 0.0   # smallest effect the per-round gate can resolve
    detectable_final: float = 0.0  # smallest effect the test/CV verdict can resolve
    recommend: str = ""            # "holdout" | "cv" (why this mode)


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def dynamic_split(
    task_ids: list[str],
    *,
    timeouts: dict[str, float] | None = None,
    seed: int = 0,
    k: int = 3,
    heavy_sec: float = 3600.0,
    kfold_threshold: int = 50,
    n_folds: int = 5,
    frac_test: float = 0.30,
    frac_audit: float = 0.10,
    frac_val: float = 0.18,
    min_test: int = 15,
    min_val: int = 8,
    min_audit: int = 4,
    min_practice: int = 10,
) -> SplitPlan:
    """Choose the four piles automatically from the benchmark **size** (and the
    per-task ``timeouts``, used only to keep the frequently-run gate fast).

    Principles:
      * **Power floors.** test/val/audit each get a minimum size, so a small
        benchmark can't yield an underpowered comparison.
      * **Small N falls back to cross-validation.** Below ``kfold_threshold`` a
        fixed held-out would be too small to be significant, so we rotate the
        test slice across ``n_folds`` and average the lift.
      * **Representative test.** The test set is reserved first from the *full*
        seeded shuffle, so it mirrors the whole benchmark (incl. heavy tasks) —
        this is the fix for the hard-unrepresentative-subset bug.
      * **Fast gate.** Pathologically slow tasks (``timeout >= heavy_sec``) are
        kept OUT of judging/audit (which run every round) and absorbed by test
        (scored once) and practice (sampled). Since runtime is ~independent of
        difficulty, this does not bias the validation difficulty distribution.
    """
    task_ids = list(task_ids)
    N = len(task_ids)
    timeouts = timeouts or {}
    order = _ordering(task_ids, seed)

    # A fixed holdout is only honest when it's both non-degenerate (every pile,
    # incl. practice, has enough tasks) AND statistically adequate (the test set
    # isn't a handful of noisy tasks). Below that, reusing tasks via k-fold CV is
    # the sound choice. The floors alone need test+val+audit+practice =
    # 15+8+4+10 = 37 tasks just to not starve a pile, and a trustworthy noisy
    # test set wants more headroom — hence kfold_threshold defaults to 50.
    floor_min = min_test + min_val + min_audit + min_practice
    use_holdout = N >= max(kfold_threshold, floor_min)

    if use_holdout:
        n_test = _clamp(round(frac_test * N), min_test, N - (min_val + min_audit + min_practice))
        n_audit = _clamp(round(frac_audit * N), min_audit, N)
        n_val = _clamp(round(frac_val * N), min_val, N)

        # 1. Reserve a REPRESENTATIVE test set from the full shuffle (incl. heavies).
        test = order[:n_test]
        rest = order[n_test:]

        # 2. Fill judging + audit from LIGHT tasks only (keep the gate fast); slow
        #    tasks fall through to practice. Fall back to heavy if light runs short.
        light = [t for t in rest if timeouts.get(t, 0.0) < heavy_sec]
        heavy = [t for t in rest if timeouts.get(t, 0.0) >= heavy_sec]
        pool = light + heavy  # light first, so heavies are the last picked
        judging = pool[:n_val]
        audit = pool[n_val:n_val + n_audit]
        practice = pool[n_val + n_audit:]

        if len(practice) >= min_practice:  # non-degenerate -> commit to holdout
            return SplitPlan(
                mode="holdout", k=k,
                split=TaskSplit(practice=practice, judging=judging, audit=audit, final_exam=test),
                rationale=(f"N={N}: test={len(test)} (representative, locked), "
                           f"judging={len(judging)} (light, k={k} gate), audit={len(audit)}, "
                           f"practice={len(practice)} (incl. {len(heavy)} heavy). "
                           f"{len([t for t in test if timeouts.get(t,0)>=heavy_sec])} heavy in test."),
            )
        # else: the floors starved practice -> fall through to k-fold.

    folds = _make_kfold(order, n_folds=min(n_folds, N), timeouts=timeouts, heavy_sec=heavy_sec)
    why = ("starves practice" if N >= kfold_threshold else f"< {kfold_threshold}")
    return SplitPlan(
        mode="kfold", k=k, folds=folds,
        rationale=(f"N={N} too small for a fixed holdout ({why}): {len(folds)}-fold "
                   f"cross-validation (rotate test slice, ~{N // max(1, len(folds))} tasks/fold), "
                   f"k={k} per task. Every task is both optimized-on and tested-on (no leak "
                   f"within a fold)."),
    )


def _make_kfold(order, *, n_folds, timeouts, heavy_sec):
    """Partition into ``n_folds`` test slices; each fold optimizes on the rest."""
    folds: list[TaskSplit] = []
    buckets = [order[i::n_folds] for i in range(n_folds)]  # round-robin = balanced
    for i in range(n_folds):
        test = buckets[i]
        rest = [t for j, b in enumerate(buckets) if j != i for t in b]
        light = [t for t in rest if timeouts.get(t, 0.0) < heavy_sec]
        heavy = [t for t in rest if timeouts.get(t, 0.0) >= heavy_sec]
        pool = light + heavy
        n_val = max(2, round(0.45 * len(rest)))
        n_audit = max(1, round(0.15 * len(rest)))
        folds.append(TaskSplit(
            practice=pool[n_val + n_audit:], judging=pool[:n_val],
            audit=pool[n_val:n_val + n_audit], final_exam=test,
        ))
    return folds


# === power-based, calibration-aware planning ===================================
#
# Validation size comes from STATISTICAL POWER + an affordability cap, so it is
# ~constant across N (an SGD mini-batch), not a fraction of N. Surplus tasks in a
# big benchmark buy test precision and a resample pool, not a bigger gate.

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


def _carve_optimization(rest: list[str], *, n_val: int, n_gen: int, n_audit: int,
                        strata: dict[str, int], timeouts: dict[str, float],
                        heavy_sec: float, seed: int) -> TaskSplit:
    """Carve judging/gen/audit/practice from optimization tasks ``rest``.

    judging+gen+audit are drawn (disjoint) from the LIGHT tasks only — they run
    every round, so a 1-2h task must never gate. They are difficulty-stratified
    for representativeness. Heavy tasks (and leftover light) fall to practice
    (sampled mini-batches, so a heavy task rarely gates failure-finding)."""
    light = [t for t in rest if timeouts.get(t, 0.0) < heavy_sec]
    heavy = [t for t in rest if timeouts.get(t, 0.0) >= heavy_sec]
    avail = list(light)
    judging = _stratified_sample(avail, n_val, strata, seed + 1)
    taken = set(judging)
    gen = _stratified_sample([t for t in avail if t not in taken], n_gen, strata, seed + 2)
    taken |= set(gen)
    audit = _stratified_sample([t for t in avail if t not in taken], n_audit, strata, seed + 3)
    taken |= set(audit)
    practice = [t for t in rest if t not in taken]  # leftover light + all heavy
    return TaskSplit(practice=practice, judging=judging, audit=audit,
                     final_exam=[], gen=gen)


def choose_eval_plan(
    task_ids: list[str],
    *,
    sigma2: float,
    difficulties: dict[str, float] | None = None,
    timeouts: dict[str, float] | None = None,
    seed: int = 0,
    k: int = 3,
    z: float = 1.96,
    delta_step: float = 0.12,
    delta_final: float = 0.05,
    val_floor: int = 8,
    val_budget_cap: int = 16,
    audit_floor: int = 4,
    test_floor: int = 15,
    practice_floor: int = 10,
    heavy_sec: float = 3600.0,
    n_folds: int = 5,
) -> SplitPlan:
    """Power-based, calibration-aware split (the 'better algorithm').

    n_val is set by statistical power (``delta_step``) clamped to a budget cap, so
    it is ~constant across N. The mode is chosen by the *detectable effect*: if a
    holdout test can't resolve ``delta_final`` (or piles would starve), fall back
    to k-fold CV. Selection is difficulty-stratified; the dual-split gate's two
    sets (``judging`` + ``gen``) are sized equally and kept off the heavy tasks.
    """
    task_ids = list(task_ids)
    N = len(task_ids)
    sigma2 = max(0.01, min(0.25, sigma2))
    timeouts = timeouts or {}
    strata = _strata(task_ids, difficulties)

    n_val = _clamp(power_n(sigma2, z=z, delta=delta_step, k=k), val_floor, val_budget_cap)
    n_gen = n_val
    n_audit = _clamp(n_val // 2, audit_floor, n_val)
    det_step = detectable_delta(n_val, sigma2, z=z, k=k)

    order = _ordering(task_ids, seed)
    # A fixed holdout needs enough tasks to (a) seat the gate piles + a modest
    # resample pool for practice, and (b) leave a test set big enough to resolve
    # delta_final. The surplus beyond the gate+practice goes to TEST (scored once,
    # so extra precision is ~free) — so test grows with N while the gate stays
    # constant. Below that, CV reuses every task instead.
    n_test_needed = power_n(sigma2, z=z, delta=delta_final, k=k)
    practice_target = max(practice_floor, 3 * n_val)
    n_test_avail = N - (n_val + n_gen + n_audit + practice_target)
    holdout_ok = n_test_avail >= max(test_floor, n_test_needed)

    if holdout_ok:
        test = _stratified_sample(order, n_test_avail, strata, seed)  # absorb surplus, representative
        rest = [t for t in order if t not in set(test)]
        s = _carve_optimization(rest, n_val=n_val, n_gen=n_gen, n_audit=n_audit,
                                strata=strata, timeouts=timeouts, heavy_sec=heavy_sec, seed=seed)
        s.final_exam = test
        det_final = detectable_delta(len(test), sigma2, z=z, k=k)
        return SplitPlan(
            mode="holdout", k=k, split=s, n_val=n_val, sigma2=sigma2,
            detectable_step=det_step, detectable_final=det_final, recommend="holdout",
            rationale=(f"N={N}: holdout test={len(test)} (representative), judging={len(s.judging)}+"
                       f"gen={len(s.gen)} (dual-split, light), audit={len(s.audit)}, "
                       f"practice={len(s.practice)}. detectable: gate~{det_step:.3f}, "
                       f"test~{det_final:.3f} (<= delta_final {delta_final})."),
        )

    # CV: rotate the test slice; each fold carves the dual-split piles from its rest.
    n_folds = min(n_folds, N) if N else 1
    buckets = [order[i::n_folds] for i in range(n_folds)]
    folds: list[TaskSplit] = []
    for i in range(n_folds):
        test = buckets[i]
        rest = [t for j, b in enumerate(buckets) if j != i for t in b]
        s = _carve_optimization(rest, n_val=n_val, n_gen=n_gen, n_audit=n_audit,
                                strata=strata, timeouts=timeouts, heavy_sec=heavy_sec, seed=seed + i)
        s.final_exam = test
        folds.append(s)
    test_total = sum(len(f.final_exam) for f in folds)  # = N (every task tested once)
    det_final = detectable_delta(test_total, sigma2, z=z, k=k)
    why = (f"holdout would need ~{n_test_needed} test tasks to resolve {delta_final} "
           f"but only {max(0, n_test_avail)} are free after the gate+practice")
    return SplitPlan(
        mode="kfold", k=k, folds=folds, n_val=n_val, sigma2=sigma2,
        detectable_step=det_step, detectable_final=det_final, recommend="cv",
        rationale=(f"N={N}: CV ({why}). {n_folds} folds (~{N // n_folds} test/fold, all {test_total} "
                   f"tested via rotation). judging={n_val}+gen={n_gen} (dual-split, light), "
                   f"audit={n_audit}. detectable: gate~{det_step:.3f}, pooled-test~{det_final:.3f}."),
    )
