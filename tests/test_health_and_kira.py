import inspect

import pytest

from studio.benchmark import kira
from studio.stages.optimize.record import health
from studio.stages.optimize.evaluate.acceptance import AcceptanceCheck
from studio.config import HealthConfig
from studio.core.state import HealthCounters


# --- health monitor ---

def test_no_signals_when_healthy():
    assert health.assess(HealthCounters(), HealthConfig()) == []


def test_acceptance_rejection_streak_signal():
    h = HealthCounters(gate_rejections=5)
    sigs = {s.name for s in health.assess(h, HealthConfig(acceptance_rejection_limit=5))}
    assert "acceptance_rejection_streak" in sigs


def test_reward_hack_signal_always_fires():
    h = HealthCounters(reward_hack_incidents=1)
    sigs = {s.name for s in health.assess(h, HealthConfig())}
    assert "reward_hack" in sigs


# --- kira harbor result parsing ---

def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_parse_harbor_results_pass_fail_and_missing(tmp_path):
    _write(tmp_path / "taskA__0/verifier/reward.txt", "1.0")   # pass
    _write(tmp_path / "taskB__0/verifier/reward.txt", "0.0")   # fail
    # taskC has no output -> 0.0
    scores = kira.parse_harbor_results(tmp_path, ["taskA", "taskB", "taskC"])
    assert scores == {"taskA": 1.0, "taskB": 0.0, "taskC": 0.0}


def test_parse_harbor_results_averages_trials(tmp_path):
    _write(tmp_path / "taskA__0/verifier/reward.txt", "1.0")
    _write(tmp_path / "taskA__1/verifier/reward.txt", "0.0")
    scores = kira.parse_harbor_results(tmp_path, ["taskA"])
    assert scores["taskA"] == 0.5


def test_complete_harbor_results_reject_missing_task(tmp_path):
    # taskA has no trials at all -> genuinely missing -> fail closed.
    _write(tmp_path / "taskB__0/verifier/reward.txt", "1.0")
    with pytest.raises(kira.BenchmarkExecutionError, match="taskA=0/2"):
        kira.require_complete_harbor_results(
            tmp_path, ["taskA", "taskB"], expected_trials=2
        )


def test_complete_harbor_results_tolerates_partial_trials(tmp_path):
    # taskA lost one of two trials -> average the one it produced, don't crash.
    _write(tmp_path / "taskA__0/verifier/reward.txt", "1.0")
    scores = kira.require_complete_harbor_results(
        tmp_path, ["taskA"], expected_trials=2
    )
    assert scores == {"taskA": 1.0}


def test_complete_harbor_results_min_trials_floor(tmp_path):
    # With min_trials=2, a single-trial task counts as missing.
    _write(tmp_path / "taskA__0/verifier/reward.txt", "1.0")
    with pytest.raises(kira.BenchmarkExecutionError, match="taskA=1/3"):
        kira.require_complete_harbor_results(
            tmp_path, ["taskA"], expected_trials=3, min_trials=2
        )


def test_kira_run_without_real_raises(tmp_path):
    from studio.core.harness import Harness
    bench = kira.KiraBenchmark(real=False)
    with pytest.raises(NotImplementedError):
        bench.run(Harness(tmp_path), ["t"])


# --- trust boundary: the acceptance must never receive a Backend ---

def test_gate_constructor_has_no_backend_param():
    params = set(inspect.signature(AcceptanceCheck.__init__).parameters)
    assert "backend" not in params
    # and the acceptance object holds no backend-like attribute
    assert not any("backend" in name for name in params)
