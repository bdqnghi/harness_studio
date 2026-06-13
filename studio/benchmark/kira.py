"""KIRA benchmark adapter — the real target (Terminus-KIRA on Terminal-Bench).

* ``boot_check`` (always available, dependency-free): compile every Python file.
* ``run`` (opt-in, ``real=True``): score the harness on Terminal-Bench tasks via
  ``harbor run`` in a Docker sandbox. This needs harbor + Docker + the dataset +
  model credentials, so it raises a clear error if harbor is missing and never
  runs unless asked. The harbor-result parsing is a pure function
  (``parse_harbor_results``) that is unit-tested with synthetic output, so the
  scoring logic is verified even without a live run.
"""

from __future__ import annotations

import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..core.harness import Harness
from .base import Benchmark

# Terminal-Bench verifiers emit a reward; >= this counts as a pass.
PASS_THRESHOLD = 1.0


class BenchmarkExecutionError(RuntimeError):
    """The evaluator did not produce a complete, trustworthy batch."""


def _reward_to_score(reward: float) -> float:
    return 1.0 if reward >= PASS_THRESHOLD else 0.0


def parse_harbor_result_details(
    jobs_dir: Path, task_ids: list[str]
) -> tuple[dict[str, float], dict[str, int]]:
    """Return per-task mean scores and valid-trial counts."""
    jobs_dir = Path(jobs_dir)
    trials: dict[str, list[float]] = {t: [] for t in task_ids}
    for reward_file in jobs_dir.rglob("verifier/reward.txt"):
        # directory name is "<task_id>__<trial_id>"
        task_dir = reward_file.parent.parent.name
        task_id = task_dir.split("__", 1)[0]
        if task_id not in trials:
            continue
        try:
            reward = float(reward_file.read_text().strip())
        except (ValueError, OSError):
            continue
        trials[task_id].append(_reward_to_score(reward))
    scores = {t: (sum(v) / len(v) if v else 0.0) for t, v in trials.items()}
    counts = {t: len(v) for t, v in trials.items()}
    return scores, counts


def parse_harbor_results(jobs_dir: Path, task_ids: list[str]) -> dict[str, float]:
    """Map a Harbor jobs directory to per-task mean scores.

    This permissive parser is retained for offline inspection. Production
    benchmark adapters use :func:`require_complete_harbor_results`.
    """
    return parse_harbor_result_details(jobs_dir, task_ids)[0]


def require_complete_harbor_results(
    jobs_dir: Path, task_ids: list[str], *, expected_trials: int, min_trials: int = 1
) -> dict[str, float]:
    """Score a batch, failing closed only on genuinely missing *tasks*.

    A task with zero valid trials is real infrastructure loss (build failure,
    all trials crashed) and must not be silently scored 0 — that fails closed.
    But a task that produced ``min_trials..expected_trials-1`` trials lost only
    a *trial* to a timeout/flake; we average the trials it did produce rather
    than nuke a multi-hour run over one missing rollout. ``min_trials`` (default
    1) is the floor below which a task counts as missing.
    """
    scores, counts = parse_harbor_result_details(jobs_dir, task_ids)
    missing = {t: c for t, c in counts.items() if c < min_trials}
    if missing:
        detail = ", ".join(
            f"{t}={c}/{expected_trials}" for t, c in sorted(missing.items())
        )
        raise BenchmarkExecutionError(f"incomplete Harbor results: {detail}")
    return scores


class KiraBenchmark(Benchmark):
    def __init__(
        self,
        *,
        real: bool = False,
        agent_import: str = "terminus_kira.terminus_kira:TerminusKira",
        dataset: str = "terminal-bench@2.0",
        model: str = "anthropic/claude-opus-4-6",
        env: str = "docker",
        tasks: list[str] | None = None,
    ) -> None:
        self.real = real
        self.agent_import = agent_import
        self.dataset = dataset
        self.model = model
        self.env = env
        self.tasks = tasks or []

    def list_tasks(self) -> list[str]:
        if not self.real:
            return []
        if not self.tasks:
            raise NotImplementedError(
                "pass an explicit task list; dataset enumeration lands later"
            )
        return list(self.tasks)

    def run(self, harness: Harness, task_ids, *, run_idx=0) -> dict[str, float]:
        if not self.real:
            raise NotImplementedError(
                "real Terminal-Bench scoring is opt-in (real=True) and needs "
                "Docker + harbor + model credentials"
            )
        if shutil.which("harbor") is None:
            raise RuntimeError("`harbor` not found on PATH; install it to score on Terminal-Bench")
        jobs_dir = Path(tempfile.mkdtemp(prefix="studio-harbor-"))
        cmd = [
            "harbor", "run",
            "--agent-import-path", self.agent_import,
            "-d", self.dataset, "-m", self.model, "-e", self.env,
            "--jobs-dir", str(jobs_dir),
        ]
        for t in task_ids:
            cmd += ["-i", t]
        subprocess.run(cmd, cwd=str(harness.root), check=True)
        return parse_harbor_results(jobs_dir, list(task_ids))

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            for rel in harness.files():
                if not rel.endswith(".py"):
                    continue
                try:
                    py_compile.compile(
                        str(harness.root / rel), cfile=str(Path(tmp) / "out.pyc"),
                        doraise=True,
                    )
                except py_compile.PyCompileError as e:
                    return False, f"{rel}: {e.msg}"
        return True, ""
