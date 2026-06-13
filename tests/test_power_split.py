"""Tests for the single 3-set split (choose_split): held_in + regression + held_out."""

import pytest

from studio.components.splitter import choose_split, detectable_delta, power_n


def _tasks(n):
    return [f"t{i:04d}" for i in range(n)]


def test_power_n_and_detectable_are_inverse_ish():
    n = power_n(0.2, z=1.96, delta=0.1, k=3)
    assert detectable_delta(n, 0.2, z=1.96, k=3) <= 0.1 + 1e-6  # n resolves delta
    # more rollouts -> fewer tasks needed; smaller effect -> more tasks
    assert power_n(0.2, delta=0.1, k=6) < power_n(0.2, delta=0.1, k=3)
    assert power_n(0.2, delta=0.05, k=3) > power_n(0.2, delta=0.10, k=3)


def test_held_in_is_constant_across_N():
    # The whole point: the optimizer's appetite (held_in) is a fixed scoop,
    # NOT a fraction of N.
    pools = {N: choose_split(_tasks(N), sigma2=0.2, seed=0).n_held_in
             for N in (200, 500, 2294, 5000)}
    assert len(set(pools.values())) == 1, f"held_in should be constant, got {pools}"
    assert next(iter(pools.values())) == 128                  # 4 * round_size(32)


def test_test_grows_with_N_while_held_in_constant():
    small = choose_split(_tasks(400), sigma2=0.2, seed=0)
    big = choose_split(_tasks(900), sigma2=0.2, seed=0)
    assert small.mode == "holdout" and big.mode == "holdout"
    assert big.n_held_out > small.n_held_out   # locked test absorbs the surplus
    assert big.n_held_in == small.n_held_in    # held-in scoop stays constant
    assert small.detectable_final <= 0.05      # a big test resolves delta_final


def test_heavy_tasks_only_in_held_out():
    # N=89 with ~15 heavy tasks -> ONE locked split; every heavy task is graded in
    # held_out and NONE may sit in an every-round set.
    tasks = _tasks(89)
    timeouts = {t: (7200.0 if i % 6 == 0 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t in tasks if timeouts[t] >= 3600}
    plan = choose_split(tasks, sigma2=0.2, timeouts=timeouts, seed=0)
    assert plan.mode == "holdout" and plan.recommend == "split"
    s = plan.split
    every_round = set(s.held_in) | set(s.regression)
    assert not (every_round & heavy), "a heavy task leaked into an every-round set"
    assert heavy <= set(s.held_out), "every heavy task must still be graded in held_out"


def test_regression_disjoint_from_held_in_and_off_heavy():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i % 10 == 0 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t in tasks if timeouts[t] >= 3600}
    s = choose_split(tasks, sigma2=0.2, timeouts=timeouts, seed=0).split
    assert not (set(s.regression) & set(s.held_in)), "regression must be independent of held_in"
    assert not (set(s.regression) & heavy)
    assert not (set(s.held_in) & heavy)


def test_stratified_test_spans_strata():
    tasks = _tasks(300)
    diff = {t: (0.1 if i % 3 == 0 else 0.5 if i % 3 == 1 else 0.9)
            for i, t in enumerate(tasks)}
    s = choose_split(tasks, sigma2=0.2, difficulties=diff, seed=1).split
    buckets = {round(diff[t], 1) for t in s.held_out}
    assert buckets == {0.1, 0.5, 0.9}, f"held_out must span all strata, got {buckets}"


def test_sigma_floor_keeps_sizing_finite():
    # All-or-nothing tasks (sigma2 -> 0) must not collapse the regression set.
    plan = choose_split(_tasks(300), sigma2=0.0, seed=0)
    assert plan.n_regression >= 16 and plan.n_held_in == 128


def test_transfer_mode_for_tiny_N():
    plan = choose_split(_tasks(20), sigma2=0.2, seed=0)
    assert plan.mode == "transfer" and plan.recommend == "transfer"


def test_deterministic_given_seed():
    a = choose_split(_tasks(300), sigma2=0.2, seed=7).split
    b = choose_split(_tasks(300), sigma2=0.2, seed=7).split
    assert a.held_out == b.held_out and a.held_in == b.held_in
    c = choose_split(_tasks(300), sigma2=0.2, seed=8).split
    assert c.held_out != a.held_out


def test_test_budget_preserves_all_heavy_tasks():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i < 20 else 600.0) for i, t in enumerate(tasks)}
    heavy = {t for t, timeout in timeouts.items() if timeout >= 3600}
    plan = choose_split(tasks, sigma2=0.2, timeouts=timeouts, test_budget_cap=50)
    assert heavy <= set(plan.split.held_out)
    assert len(plan.split.held_out) == 50


def test_test_budget_rejects_impossible_cap():
    tasks = _tasks(300)
    timeouts = {t: (7200.0 if i < 30 else 600.0) for i, t in enumerate(tasks)}
    with pytest.raises(ValueError, match="locked-test invariants"):
        choose_split(tasks, sigma2=0.2, timeouts=timeouts, test_budget_cap=25)
