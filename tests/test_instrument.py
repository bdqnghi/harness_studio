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


# --- disk-backed persistence ---


def test_disk_cache_survives_restart(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    cache_file = tmp_path / "scores.jsonl"
    tasks = ["echo-0", "echo-1"]
    first = InstrumentedBenchmark(ToyBenchmark(per_family=4), disk_path=cache_file)
    scores = first.run(h, tasks)
    assert first.task_runs == 2

    fresh = InstrumentedBenchmark(ToyBenchmark(per_family=4), disk_path=cache_file)
    again = fresh.run(h, tasks)
    assert again == scores
    assert fresh.task_runs == 0 and fresh.cache_hits == 2


def test_disk_cache_namespaced_by_config(tmp_path):
    """A k=1 acceptance score must not satisfy a k=3 lookup over the same file."""
    h = build_toy_harness(tmp_path / "h")
    cache_file = tmp_path / "scores.jsonl"

    class _K(ToyBenchmark):
        def __init__(self, k):
            super().__init__(per_family=4)
            self.k = k

    InstrumentedBenchmark(_K(1), disk_path=cache_file).run(h, ["echo-0"])
    other = InstrumentedBenchmark(_K(3), disk_path=cache_file)
    other.run(h, ["echo-0"])
    assert other.task_runs == 1 and other.cache_hits == 0


def test_disk_cache_skips_malformed_and_out_of_range(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    cache_file = tmp_path / "scores.jsonl"
    seeded = InstrumentedBenchmark(ToyBenchmark(per_family=4), disk_path=cache_file)
    seeded.run(h, ["echo-0"])
    ns = seeded.namespace
    hh = h.content_hash()
    with open(cache_file, "a") as f:
        f.write("not json at all\n")
        f.write(f'{{"ns": "{ns}", "h": "{hh}", "r": 0, "t": "echo-1", "s": 7.0}}\n')

    fresh = InstrumentedBenchmark(ToyBenchmark(per_family=4), disk_path=cache_file)
    fresh.run(h, ["echo-0", "echo-1"])
    # echo-0 came from disk; the out-of-range echo-1 record was rejected and re-run.
    assert fresh.cache_hits == 1 and fresh.task_runs == 1


def test_disk_cache_ignored_when_cache_off(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    cache_file = tmp_path / "scores.jsonl"
    InstrumentedBenchmark(ToyBenchmark(per_family=4), disk_path=cache_file).run(h, ["echo-0"])
    off = InstrumentedBenchmark(ToyBenchmark(per_family=4), cache=False, disk_path=cache_file)
    off.run(h, ["echo-0"])
    assert off.task_runs == 1 and off.cache_hits == 0
