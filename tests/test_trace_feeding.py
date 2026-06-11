"""Trace-feeding: the Diagnoser/Strategist see WHY a task failed (verifier output
+ agent trajectory), not just the task name — the lever that closes the gap with
AHE's trace/Agent-Debugger analysis. Must degrade gracefully (no trace -> "")."""

import json

import pytest

from studio.benchmark.base import Benchmark
from studio.benchmark.nexau import NexauBenchmark
from studio.components import diagnoser, runner


def _make_trial(jobs_dir, task, verifier_text, messages):
    trial = jobs_dir / "2026-01-01__00-00-00" / f"{task}__abc123"
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("0.0")
    (trial / "verifier" / "test-stdout.txt").write_text(verifier_text)
    (trial / "agent" / "nexau_in_memory_tracer.cleaned.json").write_text(
        json.dumps({"messages": messages, "output": "done"})
    )
    return trial


def test_nexau_last_trace_extracts_verifier_and_trajectory(tmp_path):
    bench = NexauBenchmark(real=False)
    jobs = tmp_path / "jobs"
    _make_trial(jobs, "write-compressor",
                "E FileNotFoundError: /app/out.txt\n2 failed",
                [{"role": "assistant", "content": "I will run gzip"},
                 {"role": "tool", "content": "gzip: command not found"}])
    bench._capture_traces(jobs, ["write-compressor"])
    trace = bench.last_trace("write-compressor")
    assert "FileNotFoundError" in trace          # verifier signal
    assert "gzip: command not found" in trace     # agent trajectory signal
    assert len(trace) <= 2400


def test_nexau_last_trace_absent_is_graceful():
    bench = NexauBenchmark(real=False)
    assert bench.last_trace("never-ran") == ""    # no trial indexed -> ""


class _StubBench(Benchmark):
    def list_tasks(self):
        return ["t1"]

    def run(self, harness, task_ids, *, run_idx=0):
        return {t: 0.0 for t in task_ids}         # all fail

    def last_trace(self, task_id):
        return f"EVIDENCE for {task_id}"


def test_runner_populates_trace(tmp_path):
    from studio.harness import Harness

    h = Harness(tmp_path / "h")
    (tmp_path / "h").mkdir()
    report = runner.run_practice(_StubBench(), h, ["t1"])
    assert report.failures and report.failures[0].trace == "EVIDENCE for t1"


class _CaptureBackend:
    def __init__(self):
        self.prompt = None

    def prompt_json(self, prompt, schema, *, tag="", model=None):
        self.prompt = prompt
        return [{"pattern_id": "p1", "description": "d", "root_cause": "r",
                 "failing_task_ids": ["t1"], "blamed_part": "tool_code", "confidence": 0.5}]


def test_diagnoser_includes_trace_evidence():
    be = _CaptureBackend()
    fails = [runner.Failure("t1", "task one", trace="verifier: AssertionError boom")]
    diagnoser.diagnose(be, fails)
    assert "failure evidence" in be.prompt
    assert "AssertionError boom" in be.prompt


def test_run_executes_full_cleanup_path(tmp_path, monkeypatch):
    """Exercise NexauBenchmark.run() end-to-end with a faked harbor so the
    parse + trace-capture + shutil.rmtree cleanup path is covered (the unit tests
    only hit _capture_traces directly, which let a missing `import shutil` slip
    into run() unnoticed)."""
    from pathlib import Path

    from studio.harness import Harness

    (tmp_path / "harbor").write_text("")  # harbor_bin must .exists()
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "code_agent.yaml").write_text("name: x\n")
    bench = NexauBenchmark(real=True, harbor_bin=tmp_path / "harbor")
    monkeypatch.setattr(bench, "_link_dataset",
                        lambda task_ids, dest: dest.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(bench, "_subprocess_env", lambda: {})
    work_dirs = []

    def fake_run(cmd, **kw):
        jobs = Path(cmd[cmd.index("--jobs-dir") + 1])
        work_dirs.append(jobs.parent)
        trial = jobs / "ts" / "t1__abc" / "verifier"
        trial.mkdir(parents=True)
        (trial / "reward.txt").write_text("1.0")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr("studio.benchmark.nexau.subprocess.run", fake_run)
    scores = bench.run(Harness(tmp_path / "h"), ["t1"], run_idx=0)
    assert scores == {"t1": 1.0}
    assert work_dirs and not work_dirs[0].exists()  # cleanup ran (no NameError)


def test_run_raises_on_harbor_failure(tmp_path, monkeypatch):
    from studio.benchmark.kira import BenchmarkExecutionError
    from studio.harness import Harness

    (tmp_path / "harbor").write_text("")
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "code_agent.yaml").write_text("name: x\n")
    bench = NexauBenchmark(real=True, harbor_bin=tmp_path / "harbor")
    monkeypatch.setattr(
        bench, "_link_dataset",
        lambda task_ids, dest: dest.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(bench, "_subprocess_env", lambda: {})

    class Failed:
        returncode = 2

    monkeypatch.setattr(
        "studio.benchmark.nexau.subprocess.run", lambda *a, **kw: Failed()
    )
    with pytest.raises(BenchmarkExecutionError, match="rc=2"):
        bench.run(Harness(tmp_path / "h"), ["t1"], run_idx=0)
