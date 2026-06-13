"""Instrumented benchmark wrapper: caching, cost counters, reward-hack defense.

Task runs are the expensive resource (PRD §4, §8), so we (a) cache scores by
``(harness_hash, run_idx, task)`` — the judging set is stable within a segment, so
re-scoring the unchanged old harness each round hits the cache — and (b) count the
task runs actually executed, to report cost-per-point (PRD §9.2).

It is also the chokepoint for the reward-hacking defense (PRD §3, §7): any score
outside [0, 1] is impossible from an honest evaluator, so we raise immediately and
let the orchestrator halt.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.harness import Harness
from .base import Benchmark


class RewardHackError(RuntimeError):
    """An impossible score was observed — the run halts (PRD §7)."""


class InstrumentedBenchmark(Benchmark):
    def __init__(
        self,
        inner: Benchmark,
        *,
        cache: bool = True,
        disk_path: Path | str | None = None,
        namespace: str | None = None,
    ) -> None:
        self.inner = inner
        self.cache_enabled = cache
        self._cache: dict[tuple[str, int, str], float] = {}
        self.task_runs = 0  # task-score evaluations actually executed
        self.cache_hits = 0
        # Optional disk persistence: one shared JSONL may back several wrappers
        # (acceptance at k=1, calibration/verdict at k=3, different actor models), so
        # every record carries a namespace encoding the evaluation config — a
        # k=1 acceptance score must never satisfy a k=3 verdict lookup.
        self.namespace = namespace or self._default_namespace()
        self.disk_path = Path(disk_path) if disk_path else None
        if self.disk_path and self.cache_enabled:
            self._load_disk()

    def _default_namespace(self) -> str:
        inner = self.inner
        return ":".join(
            str(p)
            for p in (
                type(inner).__name__,
                f"k={getattr(inner, 'k', 1)}",
                f"model={getattr(inner, 'model', '')}",
                f"tm={getattr(inner, 'timeout_multiplier', '')}",
            )
        )

    def _load_disk(self) -> None:
        try:
            lines = self.disk_path.read_text().splitlines()
        except OSError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("ns") != self.namespace:
                    continue
                score = float(rec["s"])
            except (ValueError, KeyError, TypeError):
                continue  # malformed line: ignore, the task will be re-run honestly
            if 0.0 <= score <= 1.0:  # the reward-hack guard applies to disk content
                self._cache[(str(rec["h"]), int(rec["r"]), str(rec["t"]))] = score

    def _append_disk(self, key: tuple[str, int, str], score: float) -> None:
        if not self.disk_path:
            return
        rec = {"ns": self.namespace, "h": key[0], "r": key[1], "t": key[2], "s": score}
        try:
            self.disk_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.disk_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass  # persistence is best-effort; the in-memory cache stays correct

    def list_tasks(self) -> list[str]:
        return self.inner.list_tasks()

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        return self.inner.boot_check(harness)

    def describe(self, task_id: str) -> str:
        return self.inner.describe(task_id)

    def last_trace(self, task_id: str, *, harness: Harness | None = None) -> str:
        return self.inner.last_trace(task_id, harness=harness)

    def last_evidence(self, task_id: str, *, harness: Harness | None = None):
        return self.inner.last_evidence(task_id, harness=harness)

    @property
    def evidence_store(self):
        return getattr(self.inner, "evidence_store", None)

    def run(self, harness, task_ids, *, run_idx=0):
        h = harness.content_hash()
        out: dict[str, float] = {}
        missing: list[str] = []
        for t in task_ids:
            key = (h, run_idx, t)
            if self.cache_enabled and key in self._cache:
                out[t] = self._cache[key]
                self.cache_hits += 1
            else:
                missing.append(t)

        if missing:
            fresh = self.inner.run(harness, missing, run_idx=run_idx)
            self.task_runs += len(missing)
            for t in missing:
                score = fresh[t]
                if not (0.0 <= score <= 1.0):
                    raise RewardHackError(
                        f"impossible score {score!r} for task {t!r}"
                    )
                self._cache[(h, run_idx, t)] = score
                self._append_disk((h, run_idx, t), score)
                out[t] = score
        return out
