"""Tests for the dual-split gate (Self-Harness C1+C2).

Setup: judging = `upper` tasks (baseline FAILS them), regression = `echo` tasks
(baseline PASSES them). So `enable_upper` improves judging; `regress_echo`
hurts regression. This lets us exercise every branch of the dual-split rule.
"""

from studio.benchmark import toy_fixes
from studio.benchmark.toy import ToyBenchmark, build_toy_harness
from studio.components.gate import Gate

JUDGING = ["upper-0", "upper-1"]      # baseline fails
REGRESSION = ["echo-0", "echo-1"]     # baseline passes


def _h(tmp_path, name, *fixes):
    h = build_toy_harness(tmp_path / name)
    for f in fixes:
        f(h.root)
    return h


def _gate(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    return bench, Gate(bench, JUDGING, wobble=0.0, regression_tasks=REGRESSION)


def test_improves_one_harms_none_accepts(tmp_path):
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper)  # judging up, regression unchanged
    d = gate.evaluate(old, new, additive=False)
    assert d.accept and d.gain > 0 and d.regression_gain == 0


def test_helps_judging_hurts_regression_rejected(tmp_path):
    # The exact overfit our v3 run suffered: better on the gate, worse on held-out.
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper, toy_fixes.regress_echo)
    d = gate.evaluate(old, new, additive=False)
    assert not d.accept and d.regressed and d.regression_gain < 0


def test_regresses_regression_only_rejected(tmp_path):
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.regress_echo)  # judging flat, regression down
    d = gate.evaluate(old, new, additive=False)
    assert not d.accept and d.regressed


def test_behavioral_neutral_on_both_rejected(tmp_path):
    # C2: a behavioral edit must strictly improve >=1 split, not just do no harm.
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new")  # identical -> neutral on both
    d = gate.evaluate(old, new, additive=False)
    assert not d.accept and "no strict gain" in d.reason


def test_additive_neutral_on_both_accepted(tmp_path):
    # An additive edit may be neutral on both visible splits (do-no-harm keeps it).
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new")  # neutral on both
    d = gate.evaluate(old, new, additive=True)
    assert d.accept


def test_single_split_mode_unchanged(tmp_path):
    # No regression_tasks -> legacy single-split do-no-harm path (gain>=0 accepts).
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    gate = Gate(bench, JUDGING, wobble=0.0)  # no regression set
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper)
    assert gate.evaluate(old, new).accept


# --- aggregate-accept mode (opt-in): pooled held-in gain vs per-slice do-no-harm ---

def test_aggregate_accept_captures_gain_the_dual_gate_rejects(tmp_path):
    """The exact tau2 cold-start failure: an edit helps the regression slice a
    lot but nudges judging slightly negative (within the wobble band). The
    strict per-slice dual gate REJECTS it (judging 'regressed'); aggregate mode
    ACCEPTS it because the pooled held-in gain is clearly positive."""
    from studio.benchmark.base import Benchmark
    from studio.components.gate import Gate
    from studio.harness import Harness

    class _Bench(Benchmark):
        def list_tasks(self): return ["j1", "j2", "r1", "r2"]
        def run(self, harness, task_ids, *, run_idx=0):
            new = "new" in harness.root.name
            # old: j1=1 j2=1 r1=0 r2=0 | new: j1=1 j2=0.8 (judging -0.1), r1=1 r2=1 (regression +1.0)
            old_s = {"j1": 1.0, "j2": 1.0, "r1": 0.0, "r2": 0.0}
            new_s = {"j1": 1.0, "j2": 0.8, "r1": 1.0, "r2": 1.0}
            src = new_s if new else old_s
            return {t: src[t] for t in task_ids}

    old = Harness(tmp_path / "old"); (tmp_path / "old").mkdir(); old.write_file("p", "x")
    new = Harness(tmp_path / "new"); (tmp_path / "new").mkdir(); new.write_file("p", "y")
    b = _Bench()
    judging, regression = ["j1", "j2"], ["r1", "r2"]
    WOBBLE = 0.2  # judging -0.1 sits inside the band; pooled +0.45 clears it

    # strict dual gate REJECTS: judging gain -0.1 fails per-slice non-regression.
    d_strict = Gate(b, judging, WOBBLE, regression_tasks=regression,
                    borderline_extra_runs=0).evaluate(old, new)
    assert d_strict.accept is False

    # aggregate gate ACCEPTS: pooled gain (1+0.8+1+1)/4 - (1+1+0+0)/4 = +0.45 > wobble.
    d_agg = Gate(b, judging, WOBBLE, regression_tasks=regression,
                 borderline_extra_runs=0, aggregate_accept=True).evaluate(old, new)
    assert d_agg.accept is True and d_agg.gain > WOBBLE


def test_aggregate_accept_still_rejects_real_regression(tmp_path):
    """Aggregate mode must still reject an edit that hurts the pool overall."""
    from studio.benchmark.base import Benchmark
    from studio.components.gate import Gate
    from studio.harness import Harness

    class _Bench(Benchmark):
        def list_tasks(self): return ["j1", "r1"]
        def run(self, harness, task_ids, *, run_idx=0):
            new = "new" in harness.root.name
            return {t: (0.0 if new else 1.0) for t in task_ids}  # new is worse everywhere

    old = Harness(tmp_path / "old"); (tmp_path / "old").mkdir(); old.write_file("p", "x")
    new = Harness(tmp_path / "new"); (tmp_path / "new").mkdir(); new.write_file("p", "y")
    agg = Gate(_Bench(), ["j1"], 0.0, regression_tasks=["r1"], aggregate_accept=True)
    d = agg.evaluate(old, new)
    assert d.accept is False and d.regressed is True
