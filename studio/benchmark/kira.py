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

from ..harness import Harness
from .base import Benchmark

# Terminal-Bench verifiers emit a reward; >= this counts as a pass.
PASS_THRESHOLD = 1.0


def _reward_to_score(reward: float) -> float:
    return 1.0 if reward >= PASS_THRESHOLD else 0.0


def parse_harbor_results(jobs_dir: Path, task_ids: list[str]) -> dict[str, float]:
    """Map a harbor jobs directory to {task_id: score in [0,1]}.

    Each task run writes ``<task>__<trial>/verifier/reward.txt`` (a float). A task
    with multiple trials scores the mean of its trial passes. Tasks with no output
    score 0.0 (a missing run is a failed run, not absent data).
    """
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
    return {t: (sum(v) / len(v) if v else 0.0) for t, v in trials.items()}


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
