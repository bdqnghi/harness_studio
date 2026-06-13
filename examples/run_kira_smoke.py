"""KIRA smoke test — validate the real seams on Terminus-KIRA, cheaply.

Steps (all but the last are free / a couple of cheap claude calls):
  1. Assemble a harness from Terminus-KIRA's editable files.
  2. Run the real Mapper and check it labels the prompt template as instructions
     and the agent module as tool code/descriptions.
  3. Run the structural check (compile every .py — dependency-free).
  4. Run one real Strategist -> shell -> structural-check cycle on a tiny edit.

The expensive Terminal-Bench acceptance is opt-in (--real-benchmark) and not wired
until M4, so this never spends Docker/model budget unless asked.

  python examples/run_kira_smoke.py
  python examples/run_kira_smoke.py --no-ai      # steps 1 & 3 only (no claude)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.benchmark.kira import KiraBenchmark  # noqa: E402
from studio.stages.optimize import mapper, shell, strategist, structural_check  # noqa: E402
from studio.core.harness import Harness  # noqa: E402
from studio.core.parts import PartType  # noqa: E402

KIRA = Path("/home/nghibui/codes/KIRA")
INSTRUCTIONS_FILE = "prompt-templates/terminus-kira.txt"
AGENT_FILE = "terminus_kira/terminus_kira.py"


def assemble_harness(dest: Path) -> Harness:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "terminus_kira").mkdir(parents=True)
    (dest / "prompt-templates").mkdir(parents=True)
    shutil.copy(KIRA / "terminus_kira/__init__.py", dest / "terminus_kira/__init__.py")
    shutil.copy(KIRA / AGENT_FILE, dest / AGENT_FILE)
    shutil.copy(KIRA / INSTRUCTIONS_FILE, dest / INSTRUCTIONS_FILE)
    shutil.copy(KIRA / "anthropic_caching.py", dest / "anthropic_caching.py")
    return Harness(dest)


def check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ai", action="store_true", help="skip the real claude calls")
    ap.add_argument("--real-benchmark", action="store_true", help="(M4) run Terminal-Bench")
    args = ap.parse_args()

    if not KIRA.exists():
        raise SystemExit(f"KIRA not found at {KIRA}")
    ws = Path(tempfile.mkdtemp(prefix="studio-kira-"))
    harness = assemble_harness(ws / "harness")
    print(f"workspace: {ws}\nharness files: {harness.files()}\n")

    ok = True

    print("Step: structural check (compile every .py)")
    bench = KiraBenchmark()
    boots, err = bench.boot_check(harness)
    ok &= check(f"harness compiles{'' if boots else ': ' + err}", boots)

    if args.real_benchmark:
        print("\n--real-benchmark: real Terminal-Bench scoring lands in M4 (skipped).")

    if args.no_ai:
        print("\n--no-ai: skipping Mapper + Strategist steps.")
        raise SystemExit(0 if ok else 1)

    from studio.backends.factory import make_backend
    backend = make_backend(args.model, log_dir=ws / "backend-logs")

    print("\nStep: real Mapper")
    pmap = mapper.map_harness(backend, harness, head_lines=80)
    for part in PartType:
        files = pmap.files_for(part)
        if files:
            print(f"    {part.value}: {files}")
    print(f"    do_not_touch: {pmap.do_not_touch}")
    ok &= check("prompt template -> instructions",
                INSTRUCTIONS_FILE in pmap.files_for(PartType.INSTRUCTIONS))
    ok &= check("agent module -> tool_code or tool_descriptions",
                AGENT_FILE in pmap.files_for(PartType.TOOL_CODE)
                or AGENT_FILE in pmap.files_for(PartType.TOOL_DESCRIPTIONS))

    print("\nStep: one Strategist -> shell -> structural-check cycle")
    candidate = harness.copy_to(ws / "candidate")
    backend.run_agent(
        "Add a single short clarifying sentence to the agent's system-prompt "
        "instructions file to reduce ambiguity. Make only that one minimal edit; "
        "do not touch any other file.",
        workspace=candidate.root,
        skill=strategist.load_skill(),
        tag="strategist",
    )
    sres = shell.enforce(harness, candidate, pmap, budget_per_part=3)
    print(f"    changed parts: { {p.value: f for p, f in sres.changed_parts.items()} }")
    print(f"    reverted (do-not-touch): {sres.reverted}")
    ok &= check("shell accepted the edit (within budget)", sres.ok)
    cres = structural_check.check(candidate, bench, backend=None)
    ok &= check(f"candidate still compiles{'' if cres.ok else ': ' + cres.error}", cres.ok)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
