"""Generalized benchmark wiring — make ANY benchmark hill-climbable.

A ``Target`` bundles everything benchmark-specific behind one interface so the
optimizer (orchestrator, gate, tree, splitter) stays fully benchmark-agnostic.
Wiring a new benchmark = register one ``Target`` + a thin ``Benchmark`` adapter
whose only real job is ``run(harness, task_ids) -> {task_id: score in [0,1]}``.

Two start modes:

* **warm start** — ``seed_harness()`` returns the benchmark's shipped baseline
  harness; SHO mutates it and tries to beat the published ``baseline_score``.
* **cold start** — ``seed_harness()`` returns ``None``; SHO runs the coding agent
  on an empty workspace to *generate* a runnable harness from ``cold_start_brief()``
  (see ``strategist.build_harness``) — the same engine that edits it later — and
  hill-climbs from there. This is what lets us wire a benchmark that ships no agent
  harness at all.

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
    """The task spec a benchmark declares to be cold-startable.

    It is just the *input to the coding agent*: from this, ``strategist.build_harness``
    runs the SAME coding-agent engine that edits the harness during hill-climbing —
    on an empty workspace — to generate a runnable round-0 harness from scratch
    (no templates, no skeletons). Everything here is benchmark-agnostic prose +
    tool signatures; no benchmark internals leak into the generation logic.
    """

    domain: str                         # one line: "multi-hop web-search QA"
    io_contract: str                    # how a task is presented; what an answer/action looks like
    tools: list[ToolSpec] = field(default_factory=list)
    # What the benchmark will execute — the contract the generated harness MUST
    # expose (entry file/function, or the file(s) the runtime reads). This is how
    # the agent knows what to build so the benchmark can run it.
    runner_contract: str = ""
    extra_notes: str = ""               # optional domain guidance
    seed_files: dict = field(default_factory=dict)  # optional {filename: starter content} pre-dropped


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
        self, backend, workdir: Path, *, force_cold: bool = False, validate=None
    ) -> Harness:
        """Return the round-0 harness for the optimizer.

        Warm start unless there's no seed (or ``force_cold`` to demo cold-start
        even where a baseline exists). Cold start runs the coding agent on an
        empty workspace to GENERATE a harness from the brief — the same engine
        that edits it later. ``validate`` (e.g. ``benchmark.boot_check``) lets the
        agent retry until the generated harness boots. Import is local so
        ``targets`` has no hard dependency on the coding engine."""
        seed = None if force_cold else self.seed_harness()
        if seed is not None:
            return seed.copy_to(Path(workdir) / "seed")
        if self.cold_start_brief is None:
            raise ValueError(
                f"target {self.name!r} has no seed harness and no cold_start_brief; "
                "cannot start"
            )
        from .components.strategist import build_harness

        return build_harness(
            backend, Path(workdir) / "cold_seed", self.cold_start_brief(),
            validate=validate,
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
