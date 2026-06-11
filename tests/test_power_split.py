"""Tests for power-based, calibration-aware split sizing (choose_eval_plan)."""

from studio.components.splitter import (
    choose_eval_plan, detectable_delta, power_n,
)


def _tasks(n):
    return [f"t{i:04d}" for i in range(n)]


def test_power_n_and_detectable_are_inverse_ish():
    n = power_n(0.2, z=1.96, delta=0.1, k=3)
    assert detectable_delta(n, 0.2, z=1.96, k=3) <= 0.1 + 1e-6  # n resolves delta
    # more rollouts -> fewer tasks needed; smaller effect -> more tasks
    assert power_n(0.2, delta=0.1, k=6) < power_n(0.2, delta=0.1, k=3)
    assert power_n(0.2, delta=0.05, k=3) > power_n(0.2, delta=0.10, k=3)


def test_n_val_is_constant_across_N():
    # The whole point: validation size is power+cap driven, NOT a fraction of N.
    sizes = {N: choose_eval_plan(_tasks(N), sigma2=0.2, seed=0, k=3).n_val
             for N in (100, 500, 2294, 5000)}
    assert len(set(sizes.values())) == 1, f"n_val should be constant, got {sizes}"
    assert 8 <= next(iter(sizes.values())) <= 24


def test_test_set_grows_with_N_while_gate_stays_constant():
    # Both well into the holdout regime (a 5% effect needs ~205 test tasks at k=3).
    small = choose_eval_plan(_tasks(400), sigma2=0.2, seed=0, k=3)
    big = choose_eval_plan(_tasks(900), sigma2=0.2, seed=0, k=3)
    assert small.mode == "holdout" and big.mode == "holdout"
    assert len(big.split.final_exam) > len(small.split.final_exam)   # test absorbs surplus
    assert big.n_val == small.n_val                                   # gate constant
    assert small.detectable_final <= 0.05                            # resolves delta_final


def test_tb2_like_falls_back_to_cv():
    # N=89, sigma2~0.2, k=3: a holdout test can't resolve delta_final=0.05 -> CV.
    plan = choose_eval_plan(_tasks(89), sigma2=0.2, seed=0, k=3)
    assert plan.mode == "kfold" and plan.recommend == "cv"
    assert plan.detectable_final < plan.detectable_step  # pooled CV test sharper than the gate
    # every task is a test task in exactly one fold
    union = [t for f in plan.folds for t in f.final_exam]
    assert sorted(union) == sorted(_tasks(89))


def test_dual_split_sets_are_disjoint_and_off_heavy():
    tasks = _tasks(89)
    timeouts = {t: (7200.0 if i % 8 == 0 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t in tasks if timeouts[t] >= 3600}
    plan = choose_eval_plan(tasks, sigma2=0.2, timeouts=timeouts, seed=0, k=3)
    for f in plan.folds:
        gate_tasks = set(f.judging) | set(f.gen) | set(f.audit)
        assert not (set(f.judging) & set(f.gen)), "judging and gen must be disjoint"
        assert not (gate_tasks & heavy), "no heavy task may sit in the every-round gate"
        assert len(f.gen) == len(f.judging)  # dual-split sets sized equally


def test_stratified_selection_is_representative():
    # 3 clear difficulty strata; the test set should span all three, not one.
    tasks = _tasks(300)
    diff = {t: (0.1 if i % 3 == 0 else 0.5 if i % 3 == 1 else 0.9)
            for i, t in enumerate(tasks)}
    plan = choose_eval_plan(tasks, sigma2=0.2, difficulties=diff, seed=1, k=3)
    test = plan.split.final_exam if plan.mode == "holdout" else plan.folds[0].final_exam
    buckets = {round(diff[t], 1) for t in test}
    assert buckets == {0.1, 0.5, 0.9}, f"test set must span all strata, got {buckets}"


def test_sigma_floor_keeps_sizing_finite():
    # All-or-nothing tasks (sigma2 -> 0) must not blow up n_val to the cap-or-1.
    plan = choose_eval_plan(_tasks(120), sigma2=0.0, seed=0, k=3)
    assert plan.n_val >= 8  # floored, not 1
