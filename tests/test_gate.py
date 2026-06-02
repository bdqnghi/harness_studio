from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness
from studio.benchmark import toy_fixes
from studio.components.gate import Gate

# Judging set with every family represented (2 tasks each).
JUDGING = [f"{fam}-{i}" for fam in FAMILIES for i in (0, 1)]


def _harness(tmp_path, name, *fixes):
    h = build_toy_harness(tmp_path / name)
    for fix in fixes:
        fix(h.root)
    return h


def test_accepts_clear_improvement(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = _harness(tmp_path, "old")
    new = _harness(tmp_path, "new", toy_fixes.enable_upper)
    gate = Gate(bench, JUDGING, wobble=0.0)
    d = gate.evaluate(old, new)
    assert d.accept and d.gain > 0


def test_rejects_no_change(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = _harness(tmp_path, "old")
    new = _harness(tmp_path, "new")  # identical
    d = Gate(bench, JUDGING, wobble=0.0).evaluate(old, new)
    assert not d.accept and d.gain == 0


def test_rejects_and_flags_regression(tmp_path):
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = _harness(tmp_path, "old")  # echo works
    new = _harness(tmp_path, "new", toy_fixes.regress_echo)  # echo disabled
    d = Gate(bench, JUDGING, wobble=0.0).evaluate(old, new)
    assert not d.accept and d.regressed and d.gain < 0


def test_wobble_band_blocks_tiny_gain(tmp_path):
    # A real improvement smaller than the wobble must not auto-accept; with no
    # injected noise the borderline re-runs confirm the positive gain -> accept.
    bench = ToyBenchmark(per_family=4, noise_per_mille=0)
    old = _harness(tmp_path, "old")
    new = _harness(tmp_path, "new", toy_fixes.enable_upper)
    # Set wobble above the true gain to force the borderline branch.
    gate = Gate(bench, JUDGING, wobble=1.0, borderline_extra_runs=3)
    d = gate.evaluate(old, new)
    assert d.borderline and d.accept and d.runs_used == 4
