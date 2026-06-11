"""Tests for the dual-split gate (Self-Harness C1+C2).

Setup: judging = `upper` tasks (baseline FAILS them), gen = `echo` tasks
(baseline PASSES them). So `enable_upper` improves judging; `regress_echo`
hurts gen. This lets us exercise every branch of the dual-split rule.
"""

from studio.benchmark import toy_fixes
from studio.benchmark.toy import ToyBenchmark, build_toy_harness
from studio.components.gate import Gate

JUDGING = ["upper-0", "upper-1"]   # baseline fails
GEN = ["echo-0", "echo-1"]         # baseline passes


def _h(tmp_path, name, *fixes):
    h = build_toy_harness(tmp_path / name)
    for f in fixes:
        f(h.root)
    return h


def _gate(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    return bench, Gate(bench, JUDGING, wobble=0.0, gen_tasks=GEN)


def test_improves_one_harms_none_accepts(tmp_path):
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper)  # judging up, gen unchanged
    d = gate.evaluate(old, new, additive=False)
    assert d.accept and d.gain > 0 and d.gen_gain == 0


def test_helps_judging_hurts_gen_rejected(tmp_path):
    # The exact overfit our v3 run suffered: better on the gate, worse on held-out.
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper, toy_fixes.regress_echo)
    d = gate.evaluate(old, new, additive=False)
    assert not d.accept and d.regressed and d.gen_gain < 0


def test_regresses_gen_only_rejected(tmp_path):
    _, gate = _gate(tmp_path)
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.regress_echo)  # judging flat, gen down
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
    # No gen_tasks -> legacy single-split do-no-harm path (gain>=0 accepts).
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    gate = Gate(bench, JUDGING, wobble=0.0)  # no gen
    old = _h(tmp_path, "old")
    new = _h(tmp_path, "new", toy_fixes.enable_upper)
    assert gate.evaluate(old, new).accept
