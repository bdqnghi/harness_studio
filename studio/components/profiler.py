"""Profiler: run the input harness over a benchmark once to learn per-task
difficulty (pass/fail) and collect trajectories.

The result (:class:`Profile`) drives difficulty-stratified split formation
(:func:`splitter.stratified_split`): ``held_in`` gets the *learnable* failures,
``regression`` gets the reliably-solved tasks (do-no-harm guard), and
``held_out`` is a representative locked test. The failing-task trajectories are
stashed in ``benchmark.evidence_store`` as a side effect of the run, so the
localizer can use them later — no separate trace pass needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Profile:
    """Per-task pass-rate of one harness on a benchmark (mean reward over k)."""

    pass_rate: dict[str, float] = field(default_factory=dict)
    k: int = 1
    harness_hash: str = ""

    def bin(self, task_id: str, *, solved: float = 0.8, failing: float = 0.2) -> str:
        """Classify a task by pass-rate: 'solved' | 'mixed' | 'failing'."""
        p = self.pass_rate.get(task_id, 0.0)
        if p >= solved:
            return "solved"
        if p <= failing:
            return "failing"
        return "mixed"

    def tasks_in(self, kind: str, *, solved: float = 0.8, failing: float = 0.2) -> list[str]:
        return [t for t in self.pass_rate
                if self.bin(t, solved=solved, failing=failing) == kind]

    def histogram(self, *, solved: float = 0.8, failing: float = 0.2) -> dict[str, int]:
        out = {"solved": 0, "mixed": 0, "failing": 0}
        for t in self.pass_rate:
            out[self.bin(t, solved=solved, failing=failing)] += 1
        return out

    def mean(self) -> float:
        return sum(self.pass_rate.values()) / len(self.pass_rate) if self.pass_rate else 0.0

    def to_dict(self) -> dict:
        return {"pass_rate": self.pass_rate, "k": self.k, "harness_hash": self.harness_hash}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        d = json.loads(Path(path).read_text())
        return cls(pass_rate={str(k): float(v) for k, v in d.get("pass_rate", {}).items()},
                   k=int(d.get("k", 1)), harness_hash=str(d.get("harness_hash", "")))


def profile_harness(benchmark, harness, task_ids, *, k: int = 1,
                    chunk: int = 25, on_error=None) -> Profile:
    """Run ``harness`` over ``task_ids`` (the benchmark is built at ``k`` rollouts)
    and return per-task pass-rate. Trajectories land in ``benchmark.evidence_store``
    for the failing tasks.

    Tasks are run in **chunks** so a single failing batch (a benchmark CLI rc=1,
    a flaky task) loses only that chunk rather than the whole profile — important
    for large domains. ``on_error(batch, exc)`` is called per failed chunk."""
    tasks = list(task_ids)
    scores: dict[str, float] = {}
    size = max(1, chunk)
    for i in range(0, len(tasks), size):
        batch = tasks[i:i + size]
        try:
            scores.update(benchmark.run(harness, batch, run_idx=0))
        except Exception as exc:  # noqa: BLE001 — one bad batch must not kill the profile
            if on_error is not None:
                on_error(batch, exc)
    return Profile(pass_rate=scores, k=k, harness_hash=harness.content_hash())
