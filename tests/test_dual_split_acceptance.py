"""Acceptance check: NET pooled (default) + strict-dual escape hatch.

held_in (judging) ∪ regression are pooled into ONE net decision. A real overall
lift is kept even when one slice dips a little within noise (measurement variance
dominates, so a small regression must not veto a genuine net gain); a regression
that outweighs the gain is still rejected. ``strict_dual`` restores per-slice veto.
"""

from studio.benchmark import toy_fixes
from studio.benchmark.base import Benchmark
from studio.benchmark.toy import ToyBenchmark, build_toy_harness
from studio.stages.optimize.acceptance import AcceptanceCheck
from studio.core.harness import Harness

JUDGE, REG = ["j1", "j2"], ["r1", "r2"]


def _pair(tmp_path, old_s, new_s):
    class _Bench(Benchmark):
        def list_tasks(self):
            return list(old_s)

        def run(self, harness, task_ids, *, run_idx=0):
            src = new_s if "new" in harness.root.name else old_s
            return {t: src[t] for t in task_ids}

    old = Harness(tmp_path / "old"); (tmp_path / "old").mkdir(); old.write_file("p", "x")
    new = Harness(tmp_path / "new"); (tmp_path / "new").mkdir(); new.write_file("p", "y")
    return _Bench(), old, new


def test_net_accepts_big_gain_small_regression(tmp_path):
    """THE case: the edit lifts held_in a lot and dips regression a little —
    net clearly positive, so it is ACCEPTED (not vetoed by the regression dip)."""
    b, old, new = _pair(tmp_path,
                        {"j1": 0.0, "j2": 0.0, "r1": 1.0, "r2": 1.0},
                        {"j1": 1.0, "j2": 1.0, "r1": 0.9, "r2": 0.9})
    d = AcceptanceCheck(b, JUDGE, 0.2, regression_tasks=REG, borderline_extra_runs=0).evaluate(old, new)
    # pooled gain = (3.8 - 2.0)/4 = +0.45 > noise_floor 0.2
    assert d.accept and d.gain > 0.2
    assert d.regression_gain < 0          # regression genuinely dipped — and we still accepted


def test_net_rejects_when_regression_outweighs_gain(tmp_path):
    """A regression that outweighs the gain still nets negative -> reject."""
    b, old, new = _pair(tmp_path,
                        {"j1": 0.0, "j2": 0.0, "r1": 1.0, "r2": 1.0},
                        {"j1": 0.1, "j2": 0.1, "r1": 0.0, "r2": 0.0})
    d = AcceptanceCheck(b, JUDGE, 0.2, regression_tasks=REG, borderline_extra_runs=0).evaluate(old, new)
    # pooled gain = (0.2 - 2.0)/4 = -0.45 -> reject
    assert not d.accept and d.regressed


def test_net_improves_one_harms_none_accepts(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = build_toy_harness(tmp_path / "old")
    new = build_toy_harness(tmp_path / "new"); toy_fixes.enable_upper(new.root)
    g = AcceptanceCheck(bench, ["upper-0", "upper-1"], 0.0, regression_tasks=["echo-0", "echo-1"])
    assert g.evaluate(old, new).accept


def test_additive_neutral_on_both_accepted(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = build_toy_harness(tmp_path / "old")
    new = build_toy_harness(tmp_path / "new")  # neutral on both
    g = AcceptanceCheck(bench, ["upper-0", "upper-1"], 0.0, regression_tasks=["echo-0", "echo-1"])
    assert g.evaluate(old, new, additive=True).accept


def test_single_split_mode_unchanged(tmp_path):
    # No regression set -> single-split do-no-harm path (gain >= 0 accepts).
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = build_toy_harness(tmp_path / "old")
    new = build_toy_harness(tmp_path / "new"); toy_fixes.enable_upper(new.root)
    assert AcceptanceCheck(bench, ["upper-0", "upper-1"], 0.0).evaluate(old, new).accept


def test_strict_dual_vetoes_any_regression(tmp_path):
    """The escape hatch: strict_dual=True restores per-slice veto, so the same
    big-gain-small-regression edit is REJECTED."""
    b, old, new = _pair(tmp_path,
                        {"j1": 0.0, "j2": 0.0, "r1": 1.0, "r2": 1.0},
                        {"j1": 1.0, "j2": 1.0, "r1": 0.9, "r2": 0.9})
    d = AcceptanceCheck(b, JUDGE, 0.2, regression_tasks=REG, borderline_extra_runs=0,
             strict_dual=True).evaluate(old, new)
    assert not d.accept
