"""A deterministic toy target with a *known optimum* — the smoke test's anchor.

The toy harness is a tiny agent whose editable parts are real files:

* ``instructions.txt`` — which operations are ENABLEd.
* ``tools.py``         — an ``OPS`` dict mapping op name -> function(arg)->str.
* ``config.json``      — spare flags (gives the loop something inert to leave alone).

A "task" is ``op:arg`` (e.g. ``reverse:hello``) and the toy actor computes the
answer by exec-ing ``tools.py`` and consulting the enabled list. Because the
answer is a pure function of the files, the *exact edits that fix each failure
family are known*, which is what lets the integration test assert the loop
climbs to the real optimum (not just "goes up").

Failure families and their ground-truth fix:
  * echo    — works from the start (gives a non-zero baseline).
  * reverse — ENABLEd but the function is buggy  -> fix tools.py   (blame: tool_code)
  * upper   — implemented but not ENABLEd        -> edit instructions (blame: instructions)
  * add     — neither implemented nor ENABLEd    -> add to tools.py + enable (blame: tool_code + instructions)

Noise floor (PRD §2): a correct task is flipped to "fail" deterministically based
on ``hash(harness, task, run_idx)``, modelling a flaky benchmark whose noise floor
the acceptance check must beat. Reproducible — no use of ``random``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core.harness import Harness
from ..core.parts import PartMap, PartType
from .base import Benchmark

FAMILIES = ["echo", "reverse", "upper", "add"]

# Deterministic argument bank so task sets are reproducible and disjoint.
_WORDS = [
    "hello", "world", "agent", "harness", "studio", "optimize", "gate", "wobble",
    "signal", "noise", "search", "plateau", "family", "strategy", "diagnose",
    "review", "ranker", "shell", "snapshot", "audit", "segment", "evidence",
    "frozen", "actor", "mapper", "runner", "kira", "toy", "python", "claude",
]


# --- the initial (buggy) toy harness ------------------------------------------

_INIT_TOOLS = '''\
"""Toy tool implementations. OPS maps an operation name to a function(arg)->str."""


def _echo(arg):
    return arg


def _reverse(arg):
    return arg  # BUG: should reverse the string


def _upper(arg):
    return arg.upper()


OPS = {
    "echo": _echo,
    "reverse": _reverse,
    "upper": _upper,
    # "add" is intentionally missing.
}
'''

_INIT_INSTRUCTIONS = """\
# Toy agent instructions
# Enable an operation by adding a line: ENABLE <op>
ENABLE echo
ENABLE reverse
"""

_INIT_CONFIG = {"max_steps": 10, "verbose": False}


def build_toy_harness(root: Path) -> Harness:
    """Write the initial buggy toy harness into ``root`` and return it."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "tools.py").write_text(_INIT_TOOLS)
    (root / "instructions.txt").write_text(_INIT_INSTRUCTIONS)
    (root / "config.json").write_text(json.dumps(_INIT_CONFIG, indent=2) + "\n")
    return Harness(root)


def toy_part_map() -> PartMap:
    """The known part map for the toy harness (skips the AI Mapper in tests)."""
    return PartMap(
        parts={
            PartType.INSTRUCTIONS: ["instructions.txt"],
            PartType.TOOL_CODE: ["tools.py"],
            PartType.MIDDLEWARE: ["config.json"],
            PartType.TOOL_DESCRIPTIONS: [],
            PartType.SKILLS: [],
            PartType.SUBAGENTS: [],
            PartType.MEMORY: [],
        },
        do_not_touch=[],
    )


# --- ground-truth reference solver (what a perfect harness would output) ------

def _reference(op: str, arg: str) -> str:
    if op == "echo":
        return arg
    if op == "reverse":
        return arg[::-1]
    if op == "upper":
        return arg.upper()
    if op == "add":
        a, b = arg.split(",")
        return str(int(a) + int(b))
    raise ValueError(f"unknown op {op}")


def _make_arg(family: str, i: int) -> str:
    if family == "add":
        return f"{i},{i * 2 + 1}"
    return _WORDS[i % len(_WORDS)] + str(i // len(_WORDS))


# --- the toy benchmark --------------------------------------------------------

class ToyBenchmark(Benchmark):
    """Scores the toy harness; ``per_family`` tasks per family; optional noise.

    ``noise_per_mille`` is the probability (in units of 1/1000) that a *correct*
    task is reported as failing on a given run — the injected noise floor.
    """

    def __init__(self, per_family: int = 12, noise_per_mille: int = 0) -> None:
        self.per_family = per_family
        self.noise_per_mille = noise_per_mille
        self._tasks: dict[str, tuple[str, str]] = {}
        for fam in FAMILIES:
            for i in range(per_family):
                tid = f"{fam}-{i}"
                arg = _make_arg(fam, i)
                self._tasks[tid] = (f"{fam}:{arg}", _reference(fam, arg))

    def list_tasks(self) -> list[str]:
        return list(self._tasks)

    def family_of(self, task_id: str) -> str:
        return task_id.rsplit("-", 1)[0]

    def describe(self, task_id: str) -> str:
        task_input, expected = self._tasks[task_id]
        return f"{task_id}: input {task_input!r} -> expected {expected!r}"

    def run(self, harness, task_ids, *, run_idx=0):
        ops, enabled, boot_ok = _load_actor(harness)
        h = harness.content_hash()
        scores: dict[str, float] = {}
        for tid in task_ids:
            task_input, expected = self._tasks[tid]
            correct = boot_ok and _execute(ops, enabled, task_input) == expected
            if correct and self._flip(h, tid, run_idx):
                correct = False
            scores[tid] = 1.0 if correct else 0.0
        return scores

    def boot_check(self, harness):
        try:
            _load_actor(harness)
        except Exception as e:  # noqa: BLE001 - report any boot failure verbatim
            return False, f"{type(e).__name__}: {e}"
        return True, ""

    def _flip(self, harness_hash: str, task_id: str, run_idx: int) -> bool:
        if self.noise_per_mille <= 0:
            return False
        seed = f"{harness_hash}:{task_id}:{run_idx}".encode()
        draw = int.from_bytes(hashlib.sha256(seed).digest()[:4], "big") % 1000
        return draw < self.noise_per_mille


# --- the toy actor (executes a harness) ---------------------------------------

def _load_actor(harness: Harness):
    """Return (ops_dict, enabled_set, boot_ok). Raises if the harness won't boot."""
    namespace: dict = {}
    exec(compile(harness.read_file("tools.py"), "tools.py", "exec"), namespace)
    ops = namespace.get("OPS")
    if not isinstance(ops, dict):
        raise ValueError("tools.py must define an OPS dict")
    enabled = set()
    for line in harness.read_file("instructions.txt").splitlines():
        line = line.strip()
        if line.startswith("ENABLE "):
            enabled.add(line[len("ENABLE "):].strip())
    return ops, enabled, True


def _execute(ops: dict, enabled: set, task_input: str) -> str | None:
    op, _, arg = task_input.partition(":")
    if op not in enabled or op not in ops:
        return None
    try:
        return ops[op](arg)
    except Exception:  # noqa: BLE001 - a crashing op just fails the task
        return None
