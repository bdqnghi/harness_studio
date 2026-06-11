"""Calibration: cheap per-task difficulty / noise / runtime to size eval splits.

One pass of the *baseline* harness over the task set yields each task's pass rate
``p_t`` (difficulty). ``sigma2 = mean_t p_t(1-p_t)`` is the per-task Bernoulli
variance proxy that drives **power-based** validation sizing
(``splitter.choose_eval_plan``). Per-task timeout (read from ``task.toml``) is a
free runtime proxy used to keep slow tasks out of the every-round gate, and the
``[metadata].difficulty`` field gives a cold-start stratification before any
calibration run.

The baseline pass scores double as the optimizer's baseline reference (its
"old harness" score), so calibration is not wasted compute — re-scoring the
unchanged baseline at the same ``run_idx`` is a cache hit (see
``benchmark/instrument.py``).
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..benchmark.base import Benchmark
from ..harness import Harness

DEFAULT_TASK_CACHE = Path(
    os.environ.get("HARBOR_TASK_CACHE", str(Path.home() / ".cache" / "harbor" / "tasks"))
)

# Bernoulli variance is p(1-p) ∈ [0, 0.25]. Floor keeps power-sizing finite when
# every task is a deterministic 0 or 1 (which would otherwise imply zero noise).
SIGMA2_FLOOR = 0.01
SIGMA2_CAP = 0.25
DEFAULT_RUNTIME_SEC = 600.0


@dataclass
class TaskStat:
    p: float            # difficulty: mean pass score in [0, 1]
    runtime_sec: float  # runtime proxy (task.toml agent timeout) or measured
    k: int = 1          # rollouts used to estimate p


@dataclass
class Calibration:
    stats: dict[str, TaskStat]
    sigma2: float
    model: str = ""
    baseline_hash: str = ""

    def p(self, task_id: str) -> float | None:
        s = self.stats.get(task_id)
        return s.p if s else None

    def variance(self) -> float:
        return self.sigma2

    def runtime(self, task_id: str, default: float = DEFAULT_RUNTIME_SEC) -> float:
        s = self.stats.get(task_id)
        return s.runtime_sec if s else default

    def difficulties(self) -> dict[str, float]:
        return {t: s.p for t, s in self.stats.items()}

    def timeouts(self) -> dict[str, float]:
        return {t: s.runtime_sec for t, s in self.stats.items()}

    def to_dict(self) -> dict:
        return {
            "sigma2": self.sigma2,
            "model": self.model,
            "baseline_hash": self.baseline_hash,
            "stats": {t: {"p": s.p, "runtime_sec": s.runtime_sec, "k": s.k}
                      for t, s in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        stats = {t: TaskStat(p=v["p"], runtime_sec=v.get("runtime_sec", DEFAULT_RUNTIME_SEC),
                             k=v.get("k", 1)) for t, v in d.get("stats", {}).items()}
        return cls(stats=stats, sigma2=d.get("sigma2", SIGMA2_CAP),
                   model=d.get("model", ""), baseline_hash=d.get("baseline_hash", ""))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_dict(json.loads(Path(path).read_text()))


def compute_sigma2(p_by_task: dict[str, float], *, floor: float = SIGMA2_FLOOR,
                   cap: float = SIGMA2_CAP) -> float:
    """Mean per-task Bernoulli variance, clamped so power-sizing stays finite."""
    if not p_by_task:
        return cap
    vals = [p * (1.0 - p) for p in p_by_task.values()]
    return max(floor, min(cap, sum(vals) / len(vals)))


def calibrate(
    benchmark: Benchmark,
    baseline: Harness,
    task_ids: list[str],
    *,
    k: int = 1,
    runtimes: dict[str, float] | None = None,
    model: str = "",
    run_idx: int = 0,
) -> Calibration:
    """One baseline pass → per-task difficulty + sigma2 (reused as the baseline ref)."""
    task_ids = list(task_ids)
    scores = benchmark.run(baseline, task_ids, run_idx=run_idx)  # {task: p in [0,1]}
    runtimes = runtimes or {}
    stats = {
        t: TaskStat(p=float(scores.get(t, 0.0)),
                    runtime_sec=float(runtimes.get(t, DEFAULT_RUNTIME_SEC)), k=k)
        for t in task_ids
    }
    sigma2 = compute_sigma2({t: stats[t].p for t in task_ids})
    return Calibration(stats=stats, sigma2=sigma2, model=model,
                       baseline_hash=baseline.content_hash())


# --- free task.toml metadata (no eval) -----------------------------------------

def _task_toml(cache: Path, task_id: str) -> dict:
    matches = sorted(Path(cache).glob(f"*/{task_id}/task.toml"))
    if not matches:
        return {}
    try:
        return tomllib.loads(matches[0].read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def read_task_timeouts(task_ids: list[str], *, cache: Path = DEFAULT_TASK_CACHE,
                       default: float = DEFAULT_RUNTIME_SEC) -> dict[str, float]:
    """Per-task agent timeout (runtime proxy) from task.toml; ``default`` if absent."""
    out: dict[str, float] = {}
    for t in task_ids:
        d = _task_toml(cache, t)
        to = (d.get("agent", {}).get("timeout_sec")
              or d.get("verifier", {}).get("timeout_sec") or default)
        out[t] = float(to)
    return out


_DIFF_RANK = {"easy": 0.25, "medium": 0.5, "hard": 0.8}


def read_difficulty_meta(task_ids: list[str], *, cache: Path = DEFAULT_TASK_CACHE) -> dict[str, float]:
    """Cold-start difficulty prior in [0,1] (lower = harder) from task.toml metadata.

    Returned as a pseudo *pass-rate* (easy→0.75, medium→0.5, hard→0.2) so it can
    stand in for calibration ``p_t`` in stratification before any baseline run."""
    out: dict[str, float] = {}
    for t in task_ids:
        d = _task_toml(cache, t)
        diff = str(d.get("metadata", {}).get("difficulty", "")).lower()
        if diff in _DIFF_RANK:
            out[t] = 1.0 - _DIFF_RANK[diff]  # harder -> lower pseudo pass-rate
    return out
