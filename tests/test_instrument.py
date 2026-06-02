import pytest

from studio.benchmark.base import Benchmark
from studio.benchmark.instrument import InstrumentedBenchmark, RewardHackError
from studio.benchmark.toy import ToyBenchmark, build_toy_harness


def test_cache_avoids_recomputation(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    bench = InstrumentedBenchmark(ToyBenchmark(per_family=4), cache=True)
    tasks = ["echo-0", "echo-1", "reverse-0"]
    bench.run(h, tasks)
    assert bench.task_runs == 3 and bench.cache_hits == 0
    bench.run(h, tasks)  # identical harness + tasks + run_idx -> all cached
    assert bench.task_runs == 3 and bench.cache_hits == 3


def test_cache_off_recomputes(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    bench = InstrumentedBenchmark(ToyBenchmark(per_family=4), cache=False)
    bench.run(h, ["echo-0"])
    bench.run(h, ["echo-0"])
    assert bench.task_runs == 2 and bench.cache_hits == 0


def test_different_run_idx_not_cached(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    bench = InstrumentedBenchmark(ToyBenchmark(per_family=4))
    bench.run(h, ["echo-0"], run_idx=0)
    bench.run(h, ["echo-0"], run_idx=1)
    assert bench.task_runs == 2


class _HackBench(Benchmark):
    def list_tasks(self):
        return ["t"]

    def run(self, harness, task_ids, *, run_idx=0):
        return {t: 9.0 for t in task_ids}  # impossible score


def test_reward_hack_raises(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    bench = InstrumentedBenchmark(_HackBench())
    with pytest.raises(RewardHackError):
        bench.run(h, ["t"])
