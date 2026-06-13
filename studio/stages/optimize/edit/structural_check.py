"""Structural check (PRD §5.7): the free pre-acceptance.

Apply-and-boot a candidate *before* spending any benchmark task runs: compile
every Python file, then run the benchmark's cheap boot check. This is also where
broken references surface (a removed name another file needs fails to compile).
On failure we optionally give the Strategist ONE repair attempt with the exact
error in context (PRD §11 Q6), then re-check; still broken -> drop the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

from studio.backends.base import Backend
from studio.benchmark.base import Benchmark
from studio.core.harness import Harness
from studio.stages.optimize.edit import strategist


@dataclass
class StructuralResult:
    ok: bool
    error: str = ""
    repaired: bool = False


def _compile_python(harness: Harness) -> str:
    """Return an error string for the first non-compiling .py file, else ""."""
    for rel in harness.files():
        if not rel.endswith(".py"):
            continue
        try:
            compile(harness.read_file(rel), rel, "exec")
        except SyntaxError as e:
            return f"{rel}: {type(e).__name__}: {e}"
    return ""


def _check_once(harness: Harness, benchmark: Benchmark) -> str:
    err = _compile_python(harness)
    if err:
        return err
    ok, msg = benchmark.boot_check(harness)
    return "" if ok else msg


def check(
    candidate: Harness,
    benchmark: Benchmark,
    *,
    backend: Backend | None = None,
    do_not_touch: list[str] | None = None,
    allow_repair: bool = True,
) -> StructuralResult:
    err = _check_once(candidate, benchmark)
    if not err:
        return StructuralResult(ok=True)

    if not (allow_repair and backend is not None):
        return StructuralResult(ok=False, error=err)

    # One repair attempt: hand the agent the exact error and let it fix the files.
    strategist.repair(backend, candidate.root, err, do_not_touch=do_not_touch)
    err2 = _check_once(candidate, benchmark)
    if err2:
        return StructuralResult(ok=False, error=err2, repaired=True)
    return StructuralResult(ok=True, repaired=True)
