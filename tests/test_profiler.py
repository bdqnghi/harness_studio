"""Profiler: per-task pass-rate + difficulty binning that drives stratified splits."""

from studio.benchmark.base import Benchmark
from studio.stages.profile import Profile, profile_harness
from studio.core.harness import Harness


class _Bench(Benchmark):
    def __init__(self, scores):
        self._scores = scores

    def list_tasks(self):
        return list(self._scores)

    def run(self, harness, task_ids, *, run_idx=0):
        return {t: self._scores[t] for t in task_ids}


def _h(tmp_path):
    root = tmp_path / "h"; root.mkdir(); h = Harness(root); h.write_file("p", "x")
    return h


def test_profile_harness_returns_per_task_pass_rate(tmp_path):
    bench = _Bench({"a": 1.0, "b": 0.5, "c": 0.0})
    h = _h(tmp_path)
    prof = profile_harness(bench, h, ["a", "b", "c"], k=2)
    assert prof.pass_rate == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert prof.k == 2 and prof.harness_hash == h.content_hash()


def test_bin_and_histogram():
    prof = Profile(pass_rate={"a": 1.0, "b": 0.5, "c": 0.0, "d": 0.9, "e": 0.1})
    assert prof.bin("a") == "solved" and prof.bin("d") == "solved"   # >= 0.8
    assert prof.bin("c") == "failing" and prof.bin("e") == "failing"  # <= 0.2
    assert prof.bin("b") == "mixed"
    assert prof.histogram() == {"solved": 2, "mixed": 1, "failing": 2}
    assert set(prof.tasks_in("failing")) == {"c", "e"}
    assert abs(prof.mean() - 0.5) < 1e-9


def test_profile_save_load_roundtrip(tmp_path):
    prof = Profile(pass_rate={"a": 1.0, "b": 0.0}, k=3, harness_hash="hh")
    p = tmp_path / "profile.json"
    prof.save(p)
    back = Profile.load(p)
    assert back.pass_rate == {"a": 1.0, "b": 0.0} and back.k == 3 and back.harness_hash == "hh"
