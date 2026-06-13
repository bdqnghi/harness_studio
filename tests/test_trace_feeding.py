"""Trace-feeding: the Diagnoser/Strategist see WHY a task failed (verifier output
+ agent trajectory), not just the task name — the lever that closes the gap with
AHE's trace/Agent-Debugger analysis. Must degrade gracefully (no trace -> "")."""

import json

import pytest

from studio.benchmark.base import Benchmark
from studio.benchmark.nexau import NexauBenchmark
from studio.components import diagnoser, runner
from studio.components.evidence import to_flat_excerpt


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


def _make_harness(root, content="name: x\n"):
    from studio.harness import Harness

    root.mkdir(parents=True, exist_ok=True)
    (root / "code_agent.yaml").write_text(content)
    return Harness(root)


def test_nexau_last_trace_extracts_verifier_and_trajectory(tmp_path):
    bench = NexauBenchmark(real=False)
    h = _make_harness(tmp_path / "h")
    jobs = tmp_path / "jobs"
    _make_trial(jobs, "write-compressor",
                "E FileNotFoundError: /app/out.txt\n2 failed",
                [{"role": "assistant", "content": "I will run gzip"},
                 {"role": "tool", "content": "gzip: command not found"}])
    bench._capture_traces(jobs, ["write-compressor"], h.content_hash())
    trace = bench.last_trace("write-compressor", harness=h)
    assert "FileNotFoundError" in trace          # verifier signal
    assert "gzip: command not found" in trace     # agent trajectory signal
    assert len(trace) <= 2400


def test_nexau_last_trace_absent_is_graceful(tmp_path):
    bench = NexauBenchmark(real=False)
    h = _make_harness(tmp_path / "h")
    assert bench.last_trace("never-ran", harness=h) == ""  # no trial indexed -> ""
    assert bench.last_trace("never-ran") == ""             # no harness identity -> ""


def test_traces_are_versioned_by_harness(tmp_path):
    """A candidate's gate run must never overwrite the live harness's traces —
    the exact bug where the diagnoser read a rejected candidate's trajectory."""
    bench = NexauBenchmark(real=False)
    live = _make_harness(tmp_path / "live", "name: live\n")
    cand = _make_harness(tmp_path / "cand", "name: cand\n")

    jobs_live = tmp_path / "jobs_live"
    _make_trial(jobs_live, "t1", "live verifier failure", [{"role": "tool", "content": "live tail"}])
    bench._capture_traces(jobs_live, ["t1"], live.content_hash())

    jobs_cand = tmp_path / "jobs_cand"
    _make_trial(jobs_cand, "t1", "candidate verifier failure", [{"role": "tool", "content": "cand tail"}])
    bench._capture_traces(jobs_cand, ["t1"], cand.content_hash())

    assert "live verifier failure" in bench.last_trace("t1", harness=live)
    assert "candidate verifier failure" in bench.last_trace("t1", harness=cand)
    assert "candidate" not in bench.last_trace("t1", harness=live)


def test_trace_buckets_evict_oldest_harness(tmp_path):
    bench = NexauBenchmark(real=False)
    hashes = [f"hash{i}" for i in range(bench._TRACE_HASHES + 2)]
    for i, hh in enumerate(hashes):
        jobs = tmp_path / f"jobs{i}"
        _make_trial(jobs, "t1", f"failure {i}", [])
        bench._capture_traces(jobs, ["t1"], hh)
    assert len(bench._traces) == bench._TRACE_HASHES
    assert hashes[0] not in bench._traces and hashes[-1] in bench._traces


class _StubBench(Benchmark):
    def list_tasks(self):
        return ["t1"]

    def run(self, harness, task_ids, *, run_idx=0):
        return {t: 0.0 for t in task_ids}         # all fail

    def last_trace(self, task_id, *, harness=None):
        return f"EVIDENCE for {task_id}"


def test_runner_populates_trace(tmp_path):
    from studio.harness import Harness

    h = Harness(tmp_path / "h")
    (tmp_path / "h").mkdir()
    report = runner.run_batch(_StubBench(), h, ["t1"])
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


def test_force_build_passed_on_every_invocation(tmp_path, monkeypatch):
    """--force-build must be passed on EVERY harbor invocation when enabled:
    harbor's non-force path uses the x86-only prebuilt compose image, which
    crashes agent setup on arm64 hosts (observed live). Docker layer caching
    makes the repeat builds cheap; the flag must never be skipped."""
    from pathlib import Path

    from studio.harness import Harness

    (tmp_path / "harbor").write_text("")
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "code_agent.yaml").write_text("name: x\n")
    bench = NexauBenchmark(real=True, harbor_bin=tmp_path / "harbor")
    monkeypatch.setattr(bench, "_link_dataset",
                        lambda task_ids, dest: dest.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(bench, "_subprocess_env", lambda: {})
    seen_cmds = []

    def fake_run(cmd, **kw):
        seen_cmds.append(cmd)
        jobs = Path(cmd[cmd.index("--jobs-dir") + 1])
        for t in ("t1", "t2"):
            trial = jobs / "ts" / f"{t}__abc" / "verifier"
            trial.mkdir(parents=True, exist_ok=True)
            (trial / "reward.txt").write_text("1.0")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr("studio.benchmark.nexau.subprocess.run", fake_run)
    h = Harness(tmp_path / "h")
    bench.run(h, ["t1"], run_idx=0)
    bench.run(h, ["t1"], run_idx=1)
    bench.run(h, ["t1", "t2"], run_idx=2)
    assert all("--force-build" in cmd for cmd in seen_cmds)


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


# --- Phase 7: coding adapters populate the structured evidence store ---

def test_nexau_last_evidence_is_structured(tmp_path):
    bench = NexauBenchmark(real=False)
    h = _make_harness(tmp_path / "h")
    jobs = tmp_path / "jobs"
    _make_trial(jobs, "write-compressor",
                "E FileNotFoundError: /app/out.txt\n2 failed",
                [{"role": "assistant", "content": "I will run gzip"},
                 {"role": "tool", "content": "gzip: command not found"}])
    bench._capture_traces(jobs, ["write-compressor"], h.content_hash())

    ev = bench.last_evidence("write-compressor", harness=h)
    assert ev is not None and ev.reward == 0.0
    assert any(s.kind == "test" and not s.passed for s in ev.signals)
    # the structured evidence materializes into a corpus the localizer can read
    dest = bench.evidence_store.materialize(h.content_hash(), tmp_path / "ev")
    corpus = (dest / "write-compressor.md").read_text()
    assert "FileNotFoundError" in corpus and "gzip: command not found" in corpus
    # versioned: a different harness has no evidence
    other = _make_harness(tmp_path / "h2", "name: y\n")
    assert bench.last_evidence("write-compressor", harness=other) is None


def test_mini_swe_last_evidence_is_structured(tmp_path):
    from studio.benchmark.mini_swe import MiniSweBenchmark

    bench = MiniSweBenchmark(real=False)
    h = _make_harness(tmp_path / "h")
    trial = tmp_path / "jobs" / "2026-01-01__00-00-00" / "fix-bug__abc"
    (trial / "verifier").mkdir(parents=True)
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier" / "reward.txt").write_text("0.0")
    (trial / "verifier" / "test-stdout.txt").write_text("AssertionError: expected 5 got 4")
    (trial / "agent" / "traj.json").write_text(
        json.dumps({"messages": [{"role": "assistant", "content": "patch applied wrong"}]}))
    bench._capture_traces(tmp_path / "jobs", ["fix-bug"], h.content_hash())

    ev = bench.last_evidence("fix-bug", harness=h)
    assert ev is not None and any(s.kind == "test" for s in ev.signals)
    assert "AssertionError" in to_flat_excerpt(ev)
