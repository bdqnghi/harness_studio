"""Generalized benchmark wiring — make ANY benchmark hill-climbable.

A ``Target`` bundles everything benchmark-specific behind one interface so the
optimizer (orchestrator, gate, tree, splitter) stays fully benchmark-agnostic.
Wiring a new benchmark = register one ``Target`` + a thin ``Benchmark`` adapter
whose only real job is ``run(harness, task_ids) -> {task_id: score in [0,1]}``.

Two start modes:

* **warm start** — ``seed_harness()`` returns the benchmark's shipped baseline
  harness; SHO mutates it and tries to beat the published ``baseline_score``.
* **cold start** — ``seed_harness()`` returns ``None``; SHO *synthesizes* a
  runnable harness from ``cold_start_brief()`` (see ``components.cold_start``)
  and hill-climbs from there. This is what lets us wire a benchmark that ships
  no agent harness at all (e.g. BrowseComp).

The key seam that makes this general is the existing ``Benchmark.run`` contract:
"given a harness directory and some task ids, return per-task scores." Each
adapter encapsulates HOW its agent is invoked (harbor/docker, a CLI, or our own
ReAct driver) and HOW the mutated harness is injected — none of which leaks to
the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .benchmark.base import Benchmark
from .harness import Harness
from .parts import PartMap


@dataclass
class ToolSpec:
    """One callable tool/API the cold-start agent may use."""

    name: str
    signature: str          # e.g. "search(query: str) -> list[str]"
    doc: str                # one-line description of what it does / returns


@dataclass
class ColdStartBrief:
    """The minimum a benchmark must declare to be cold-startable.

    From this, ``cold_start.bootstrap_harness`` synthesizes a runnable-but-
    unoptimized harness (system prompt + tool wiring + agent loop) that the
    optimizer then hill-climbs. Everything here is benchmark-agnostic prose +
    tool signatures — no benchmark internals leak into the synthesis logic.
    """

    domain: str                         # one line: "multi-hop web-search QA"
    io_contract: str                    # how a task is presented; what an answer/action looks like
    tools: list[ToolSpec] = field(default_factory=list)
    template: str = "react"             # which built-in skeleton: "react" | "code" | "policy"
    extra_notes: str = ""               # optional domain guidance for the initial prompt
    extra_files: dict = field(default_factory=dict)  # extra {filename: bare seed content} to write


@dataclass
class TargetConfig:
    """Per-run knobs an adapter needs to build its Benchmark (model, k, etc.).

    Deliberately small and provider-neutral; adapters read what they need."""

    model: str                          # litellm-style "provider/model" (or bare; adapter normalizes)
    provider: str | None = None
    k: int = 1                          # rollouts/task
    n_concurrent: int = 8
    real: bool = True                   # False => dry-run/plan only (no spend)
    extra: dict = field(default_factory=dict)  # adapter-specific (e.g. domain, ahe_dir, subset)


@dataclass
class Target:
    """Everything benchmark-specific, bundled. Registered in ``TARGETS``."""

    name: str
    make_benchmark: Callable[[TargetConfig], Benchmark]
    part_map: Callable[[], PartMap]
    # Warm start: returns the shipped baseline harness. Cold start: returns None.
    seed_harness: Callable[[], Harness | None] = lambda: None
    # Required iff seed_harness() can return None (cold start path).
    cold_start_brief: Callable[[], ColdStartBrief] | None = None
    # Published baseline to beat/match (the success bar); None if we set our own.
    baseline_score: float | None = None
    # Human-facing note for reports (which model/leaderboard the baseline is from).
    baseline_note: str = ""

    def resolve_seed(
        self, backend, workdir: Path, *, force_cold: bool = False
    ) -> Harness:
        """Return the round-0 harness for the optimizer.

        Warm start unless there's no seed (or ``force_cold`` to demo cold-start
        even where a baseline exists). Cold start synthesizes one from the brief.
        Import is local so ``targets`` has no hard dependency on the synthesis
        engine (and tests can stub it)."""
        seed = None if force_cold else self.seed_harness()
        if seed is not None:
            return seed.copy_to(Path(workdir) / "seed")
        if self.cold_start_brief is None:
            raise ValueError(
                f"target {self.name!r} has no seed harness and no cold_start_brief; "
                "cannot start"
            )
        from .components.cold_start import bootstrap_harness

        return bootstrap_harness(
            backend, self.cold_start_brief(), Path(workdir) / "cold_seed"
        )


# --- registry --------------------------------------------------------------

# Populated lazily (each entry imports its adapter only when built) so that
# importing ``targets`` never drags in harbor/docker/benchmark-specific deps.
_REGISTRY: dict[str, Callable[[], Target]] = {}


def register(name: str, factory: Callable[[], Target]) -> None:
    _REGISTRY[name] = factory


def get_target(name: str) -> Target:
    if name not in _REGISTRY:
        raise KeyError(f"unknown target {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def list_targets() -> list[str]:
    return sorted(_REGISTRY)
