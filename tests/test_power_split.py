"""Tests for the single 3-set split (choose_split): held-in pool + regression + test."""

from studio.components.splitter import (
    choose_split, detectable_delta, power_n,
)
import pytest


def _tasks(n):
    return [f"t{i:04d}" for i in range(n)]


def test_power_n_and_detectable_are_inverse_ish():
    n = power_n(0.2, z=1.96, delta=0.1, k=3)
    assert detectable_delta(n, 0.2, z=1.96, k=3) <= 0.1 + 1e-6  # n resolves delta
    # more rollouts -> fewer tasks needed; smaller effect -> more tasks
    assert power_n(0.2, delta=0.1, k=6) < power_n(0.2, delta=0.1, k=3)
    assert power_n(0.2, delta=0.05, k=3) > power_n(0.2, delta=0.10, k=3)


def test_held_in_is_constant_across_N():
    # The whole point: the optimizer's appetite (pool + judging) is a fixed scoop,
    # NOT a fraction of N.
    pools = {N: choose_split(_tasks(N), sigma2=0.2, seed=0).n_pool
             for N in (200, 500, 2294, 5000)}
    judg = {N: choose_split(_tasks(N), sigma2=0.2, seed=0).n_judging
            for N in (200, 500, 2294, 5000)}
    assert len(set(pools.values())) == 1, f"pool should be constant, got {pools}"
    assert next(iter(pools.values())) == 128                  # 4 * round_size(32)
    assert len(set(judg.values())) == 1, f"judging should be constant, got {judg}"


def test_test_grows_with_N_while_held_in_constant():
    small = choose_split(_tasks(400), sigma2=0.2, seed=0)
    big = choose_split(_tasks(900), sigma2=0.2, seed=0)
    assert small.mode == "holdout" and big.mode == "holdout"
    assert big.n_test > small.n_test          # locked test absorbs the surplus
    assert big.n_pool == small.n_pool          # held-in scoop stays constant
    assert small.detectable_final <= 0.05      # a big test resolves delta_final


def test_tb2_is_single_holdout_with_heavy_in_test():
    # N=89 with ~15 heavy tasks -> ONE locked split; every heavy task is graded in
    # the test set and NONE may sit in any every-round set.
    tasks = _tasks(89)
    timeouts = {t: (7200.0 if i % 6 == 0 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t in tasks if timeouts[t] >= 3600}
    plan = choose_split(tasks, sigma2=0.2, timeouts=timeouts, seed=0)
    assert plan.mode == "holdout" and plan.recommend == "split"
    s = plan.split
    every_round = set(s.judging) | set(s.regression) | set(s.audit) | set(s.practice)
    assert not (every_round & heavy), "a heavy task leaked into an every-round set"
    assert heavy <= set(s.final_exam), "every heavy task must still be graded in test"


def test_regression_disjoint_from_pool_and_gate_off_heavy():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i % 10 == 0 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t in tasks if timeouts[t] >= 3600}
    s = choose_split(tasks, sigma2=0.2, timeouts=timeouts, seed=0).split
    assert not (set(s.regression) & set(s.practice)), "regression must be independent of the pool"
    assert not (set(s.regression) & heavy)
    assert not (set(s.judging) & heavy)
    assert set(s.judging) <= set(s.practice), "the gate slice is drawn from the held-in pool"


def test_stratified_test_spans_strata():
    # 3 clear difficulty strata; the locked test should span all three.
    tasks = _tasks(300)
    diff = {t: (0.1 if i % 3 == 0 else 0.5 if i % 3 == 1 else 0.9)
            for i, t in enumerate(tasks)}
    s = choose_split(tasks, sigma2=0.2, difficulties=diff, seed=1).split
    buckets = {round(diff[t], 1) for t in s.final_exam}
    assert buckets == {0.1, 0.5, 0.9}, f"test set must span all strata, got {buckets}"


def test_sigma_floor_keeps_sizing_finite():
    # All-or-nothing tasks (sigma2 -> 0) must not collapse the gate below its floor.
    plan = choose_split(_tasks(300), sigma2=0.0, seed=0)
    assert plan.n_judging >= 8


def test_transfer_mode_for_tiny_N():
    # Too few tasks to lock an honest held-out -> transfer mode.
    plan = choose_split(_tasks(20), sigma2=0.2, seed=0)
    assert plan.mode == "transfer" and plan.recommend == "transfer"


def test_deterministic_given_seed():
    a = choose_split(_tasks(300), sigma2=0.2, seed=7).split
    b = choose_split(_tasks(300), sigma2=0.2, seed=7).split
    assert a.final_exam == b.final_exam and a.judging == b.judging
    # different seed -> different locked test (don't always freeze the same tasks)
    c = choose_split(_tasks(300), sigma2=0.2, seed=8).split
    assert c.final_exam != a.final_exam


def test_test_budget_preserves_all_heavy_tasks():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i < 20 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t, timeout in timeouts.items() if timeout >= 3600}
    plan = choose_split(
        tasks, sigma2=0.2, timeouts=timeouts, test_budget_cap=50
    )
    assert heavy <= set(plan.split.final_exam)
    assert len(plan.split.final_exam) == 50


def test_test_budget_rejects_impossible_cap():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i < 30 else 600.0) for i, t in enumerate(tasks)}
    with pytest.raises(ValueError, match="locked-test invariants"):
        choose_split(
            tasks, sigma2=0.2, timeouts=timeouts, test_budget_cap=25
        )
