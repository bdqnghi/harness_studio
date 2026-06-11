"""Tests for the calibration module (difficulty / sigma2 / task.toml readers)."""

from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness
from studio.components.calibration import (
    Calibration, calibrate, compute_sigma2, read_difficulty_meta, read_task_timeouts,
)

TASKS = [f"{f}-{i}" for f in FAMILIES for i in (0, 1)]


def test_compute_sigma2_bounds():
    assert compute_sigma2({"a": 0.5, "b": 0.5}) == 0.25          # max variance at p=0.5
    assert compute_sigma2({"a": 1.0, "b": 0.0}) == 0.01          # floored (not 0)
    assert compute_sigma2({}) == 0.25                            # empty -> conservative cap


def test_compute_sigma2_corrects_finite_rollout_rates():
    raw = compute_sigma2({"a": 0.2, "b": 0.8})
    corrected = compute_sigma2({"a": 0.2, "b": 0.8}, k=3)
    assert corrected > raw


def test_calibrate_records_difficulty_and_sigma(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    base = build_toy_harness(tmp_path / "base")
    cal = calibrate(bench, base, TASKS, k=1, model="toy")
    # baseline: echo passes, others fail -> p in {0,1}; sigma2 floored.
    assert set(cal.difficulties()) == set(TASKS)
    assert cal.sigma2 == 0.01
    assert cal.model == "toy" and cal.baseline_hash
    # the per-task p IS the baseline score (reused as the reference)
    raw = bench.run(base, TASKS, run_idx=0)
    assert cal.difficulties() == raw


def test_calibration_roundtrip(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    cal = calibrate(bench, build_toy_harness(tmp_path / "b"), TASKS)
    p = tmp_path / "cal.json"
    cal.save(p)
    back = Calibration.load(p)
    assert back.sigma2 == cal.sigma2 and back.difficulties() == cal.difficulties()


def _write_task(cache, task_id, *, difficulty="medium", agent_timeout=900.0):
    d = cache / "hash0" / task_id
    d.mkdir(parents=True)
    (d / "task.toml").write_text(
        f'[metadata]\ndifficulty = "{difficulty}"\n[agent]\ntimeout_sec = {agent_timeout}\n'
    )


def test_read_task_timeouts_and_difficulty(tmp_path):
    cache = tmp_path / "tasks"
    _write_task(cache, "fast-easy", difficulty="easy", agent_timeout=600.0)
    _write_task(cache, "slow-hard", difficulty="hard", agent_timeout=7200.0)
    tos = read_task_timeouts(["fast-easy", "slow-hard", "missing"], cache=cache, default=300.0)
    assert tos == {"fast-easy": 600.0, "slow-hard": 7200.0, "missing": 300.0}
    diffs = read_difficulty_meta(["fast-easy", "slow-hard", "missing"], cache=cache)
    # easy -> high pseudo pass-rate, hard -> low; missing absent
    assert diffs["fast-easy"] > diffs["slow-hard"]
    assert "missing" not in diffs
