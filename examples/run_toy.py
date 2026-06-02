"""Run the optimizer end-to-end on the toy target.

  python examples/run_toy.py --backend mock     # deterministic, free
  python examples/run_toy.py --backend claude    # real claude -p Strategist

The mock backend follows a fixed proposer script (good fixes + bad edits) so the
run is reproducible; the claude backend lets a real coding agent try to improve
the toy harness on its own.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.backends.mock import MockBackend  # noqa: E402
from studio.benchmark import toy_fixes  # noqa: E402
from studio.benchmark.toy import (  # noqa: E402
    FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map,
)
from studio.components.splitter import TaskSplit  # noqa: E402
from studio.config import Config, EditConfig, LoopConfig  # noqa: E402
from studio.orchestrator import Orchestrator  # noqa: E402

SPLIT = TaskSplit(
    judging=[f"{f}-{i}" for f in FAMILIES for i in (0, 1)],
    final_exam=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
    audit=[f"{f}-{i}" for f in FAMILIES for i in (4, 5)],
    practice=[f"{f}-{i}" for f in FAMILIES for i in range(6, 12)],
)

# Mock proposer script: per round a *losing* first strategy and a *winning*
# second one, so the demo shows competing strategies + gate fall-through.
MOCK_ROUNDS = 3
MOCK_STRATEGIES = 2
MOCK_ACTIONS = [
    toy_fixes.enable_bogus, toy_fixes.enable_upper,   # round 1
    toy_fixes.break_boot, toy_fixes.fix_reverse,      # round 2
    toy_fixes.regress_echo, toy_fixes.fix_add_full,   # round 3
]
MOCK_DIAGNOSIS = [{
    "pattern_id": "p1", "description": "several operations fail",
    "root_cause": "buggy or disabled ops", "failing_task_ids": ["reverse-0"],
    "blamed_part": "tool_code", "confidence": 0.7,
}]


def make_backend(name: str):
    if name == "mock":
        return MockBackend(
            json_responses={
                "diagnoser": [MOCK_DIAGNOSIS] * MOCK_ROUNDS,
                "reviewer": [{"keep": [], "drop": []}] * MOCK_ROUNDS,
                "ranker": [{"order": []}] * MOCK_ROUNDS,
            },
            agent_actions={"strategist": list(MOCK_ACTIONS)},
        )
    if name == "claude":
        from studio.backends.claude_cli import ClaudeCLIBackend
        return ClaudeCLIBackend()
    raise SystemExit(f"unknown backend {name!r}")


def family_scores(bench: ToyBenchmark, harness, tasks) -> dict[str, float]:
    scores = bench.run(harness, tasks, run_idx=0)
    out: dict[str, list[float]] = {f: [] for f in FAMILIES}
    for tid, s in scores.items():
        out[bench.family_of(tid)].append(s)
    return {f: (sum(v) / len(v) if v else 0.0) for f, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "claude"])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--strategies", type=int, default=None, help="strategies/round (claude)")
    ap.add_argument("--noise", type=int, default=0, help="injected wobble per-mille")
    ap.add_argument("--workspace", default=None)
    args = ap.parse_args()

    is_mock = args.backend == "mock"
    rounds = args.rounds or (MOCK_ROUNDS if is_mock else 4)
    n_strategies = MOCK_STRATEGIES if is_mock else (args.strategies or 3)
    bench = ToyBenchmark(per_family=12, noise_per_mille=args.noise)
    ws = Path(args.workspace or tempfile.mkdtemp(prefix="studio-toy-"))
    src = build_toy_harness(ws / "src")

    orch = Orchestrator(
        workspace=ws / "run",
        source_harness=src,
        benchmark=bench,
        backend=make_backend(args.backend),
        config=Config(
            noise_per_mille=args.noise,
            loop=LoopConfig(
                rounds=rounds, wobble_runs=3, strategies_per_round=n_strategies,
                # small segments for the mock so the meta-loop boundary is visible
                segment_length=2 if is_mock else 10,
            ),
            # the mock's break_boot edit is intentionally unrepairable
            edits=EditConfig(allow_repair=not is_mock),
        ),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    before = family_scores(bench, orch.harness, SPLIT.final_exam)
    result = orch.run()
    after = family_scores(bench, orch.harness, SPLIT.final_exam)

    print(f"\nworkspace: {ws}")
    print(f"wobble (noise floor): {result.wobble:.3f}")
    print("\nround  accepted  gain     old -> new   note")
    for r in result.rounds:
        print(f"{r.round_idx:>4}  {str(r.accepted):>8}  {r.gain:+.3f}  "
              f"{r.old_score:.2f}->{r.new_score:.2f}   {r.note}")

    print(f"\nfinal-exam: baseline {result.baseline_final:.3f} -> "
          f"final {result.final_score:.3f}  (uplift {result.uplift:+.3f})")
    print("per-family final-exam score (before -> after):")
    for f in FAMILIES:
        print(f"  {f:<8} {before[f]:.2f} -> {after[f]:.2f}")
    print(f"\naccepted {result.accepted}/{len(result.rounds)} rounds")
    cpp = "n/a" if result.cost_per_point == float("inf") else f"{result.cost_per_point:.1f}"
    print(f"cost: {result.task_runs} task-runs, {result.cache_hits} cache hits, "
          f"{cpp} task-runs/point")
    if result.halted:
        print("RUN HALTED (reward-hacking incident)")
    if result.family_map and (result.family_map.works or result.family_map.falsified
                              or result.family_map.pivot):
        print("\nstrategy-family map learned:")
        for label, items in (("works", result.family_map.works),
                             ("falsified", result.family_map.falsified),
                             ("pivot", result.family_map.pivot)):
            for it in items:
                print(f"  [{label}] {it}")


if __name__ == "__main__":
    main()
