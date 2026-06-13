"""Pipeline stages: split routing (stratified vs random) + the verdict math."""

from types import SimpleNamespace

from studio import pipeline
from studio.stages.profile import Profile


def _args(**kw):
    base = dict(seed=0, held_in=4, reg=2, held_out=4, no_profile=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_split_random_when_no_profile():
    sp = pipeline.build_split(_args(no_profile=True), profile=None,
                              tasks=[f"t{i}" for i in range(20)])
    assert len(sp.held_in) == 4 and len(sp.regression) == 2 and len(sp.held_out) == 4


def test_build_split_stratified_from_profile():
    prof = Profile(pass_rate={**{f"f{i}": 0.0 for i in range(6)},
                              **{f"s{i}": 1.0 for i in range(6)}})
    sp = pipeline.build_split(_args(), profile=prof, tasks=list(prof.pass_rate))
    assert all(t.startswith("f") for t in sp.held_in)      # learnable failures
    assert all(t.startswith("s") for t in sp.regression)   # do-no-harm guard


def test_verdict_math():
    class _Bench:
        def run(self, harness, tasks, *, run_idx=0):
            return {t: (1.0 if getattr(harness, "new", False) else 0.0) for t in tasks}

    class _H:
        def __init__(self, new):
            self.new = new

    res = pipeline.verdict(_Bench(), _H(False), _H(True), ["a", "b"], k=3, sigma2=0.2)
    assert res["baseline_harness_score"] == 0.0
    assert res["optimized_harness_score"] == 1.0
    assert res["lift"] == 1.0
    assert res["detectable"] > 0
