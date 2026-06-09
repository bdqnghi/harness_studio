"""Optimize AHE's *exact* input harness on Terminal-Bench 2 — the head-to-head.

This drives harness_studio's full optimizer over the **same** NexAU harness AHE
evolves (``agents/code_agent_simple/``), scored by the **same** harbor path
(``harbor run --agent nexau``) with the **same** actor model (gpt-5.4). The only
difference vs AHE is the optimizer producing the workspace — which is the whole
point of the comparison.

Usage
-----
    # Free: print the plan + the exact harbor command, spend nothing.
    python examples/run_nexau_tb2.py --dry-run

    # Real (needs Docker + AHE's harbor + gpt-5.4 creds in AHE/.env):
    python examples/run_nexau_tb2.py --tasks fix-git,sqlite-db-truncate,... \
        --rounds 4 --segment-length 2 --strategies 2 \
        --final 2 --audit 2 --judging 2

The four piles are carved from ``--tasks`` (final_exam locked, then audit,
judging, practice = remainder). Keep them small: every gate round scores the
judging set on both the old and new harness, and each task-eval is ~10-45 min
on emulated Docker, so cost grows fast.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.tb2_config import PILES, SEED, TASKS  # noqa: E402
from studio.benchmark.nexau import DEFAULT_AHE_DIR, NexauBenchmark  # noqa: E402
from studio.components.splitter import split_tasks  # noqa: E402
from studio.config import Config, EditConfig, LoopConfig, PileConfig  # noqa: E402
from studio.harness import Harness  # noqa: E402
from studio.parts import PartMap, PartType  # noqa: E402

CODE_AGENT_SIMPLE = DEFAULT_AHE_DIR / "agents" / "code_agent_simple"


def nexau_part_map() -> PartMap:
    """Explicit, fair search space for the NexAU harness: AHE-equivalent freedom.

    Directory entries (trailing ``/``) let the optimizer ADD new tools, middleware,
    skills, and sub-agents (not just edit existing files), and ``code_agent.yaml``
    is editable so new components can be registered. Only the frozen actor
    (``llm_config``, protected by the Strategist skill + the env model reference)
    and inert infra files are off-limits — mirroring exactly what AHE may change.
    """
    return PartMap(
        parts={
            PartType.INSTRUCTIONS: ["systemprompt.md"],
            PartType.TOOL_DESCRIPTIONS: ["tool_descriptions/"],
            PartType.TOOL_CODE: ["tools/"],
            PartType.MIDDLEWARE: ["middleware/"],
            PartType.SKILLS: ["skills/", ".claude/"],
            PartType.SUBAGENTS: ["code_agent.yaml", "sub_agents/"],
            PartType.MEMORY: ["LongTermMEMORY.md"],
        },
        do_not_touch=["nexau.json", "start.py", "README.md", ".gitignore", "ShortTermMEMORY.md"],
    )

# The fixed head-to-head task set (shared with the AHE arm via tb2_config) so the
# held-out final pile is identical for both. Override with --tasks for ad-hoc runs.
DEFAULT_TASKS = TASKS


def build(args) -> tuple[Harness, NexauBenchmark, Config, object]:
    src = Harness(CODE_AGENT_SIMPLE)
    bench = NexauBenchmark(
        real=not args.dry_run,
        ahe_dir=args.ahe_dir,
        tasks=args.tasks,
        model=args.model,
        env=args.env,
        n_concurrent=args.n_concurrent,
        k=args.k,
        timeout_multiplier=args.timeout_multiplier,
    )
    piles = PileConfig(
        practice=max(1, len(args.tasks) - args.final - args.audit - args.judging),
        judging=args.judging,
        audit=args.audit,
        final_exam=args.final,
    )
    cfg = Config(
        seed=args.seed,
        cache=True,
        piles=piles,
        loop=LoopConfig(
            rounds=args.rounds,
            segment_length=args.segment_length,
            wobble_runs=args.wobble_runs,
            strategies_per_round=args.strategies,
        ),
        edits=EditConfig(budget_per_part=args.budget),
    )
    backend = None
    if not args.dry_run:
        from studio.backends.claude_cli import ClaudeCLIBackend

        kw = {"log_dir": Path(args.workspace) / "claude-logs"}
        if args.proposer_model:
            kw["tier_a_model"] = args.proposer_model
        backend = ClaudeCLIBackend(**kw)
    return src, bench, cfg, backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print plan + harbor cmd; spend nothing")
    ap.add_argument("--tasks", type=lambda s: [t.strip() for t in s.split(",") if t.strip()],
                    default=DEFAULT_TASKS, help="comma-separated TB2 task names")
    ap.add_argument("--ahe-dir", type=Path, default=DEFAULT_AHE_DIR)
    ap.add_argument("--workspace", type=Path, default=None)
    ap.add_argument("--model", default="gpt-5.4", help="actor model (must match AHE for fairness)")
    ap.add_argument("--proposer-model", default=None, help="Tier-A Strategist model (default: backend default)")
    ap.add_argument("--env", default="docker")
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument("--k", type=int, default=1, help="rollouts per task per harbor call")
    ap.add_argument("--timeout-multiplier", type=float, default=3.0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--segment-length", type=int, default=2, help="< rounds so the meta-loop fires")
    ap.add_argument("--strategies", type=int, default=2)
    ap.add_argument("--wobble-runs", type=int, default=2)
    ap.add_argument("--final", type=int, default=PILES.final_exam)
    ap.add_argument("--audit", type=int, default=PILES.audit)
    ap.add_argument("--judging", type=int, default=PILES.judging)
    ap.add_argument("--budget", type=int, default=4, help="max changed files per part per strategy")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.workspace is None:
        args.workspace = Path(tempfile.mkdtemp(prefix="studio-nexau-tb2-"))

    src, bench, cfg, backend = build(args)

    # Show the split + the exact harbor command (free).
    split = split_tasks(args.tasks, cfg.piles, seed=cfg.seed)
    print("=== harness_studio vs AHE: same NexAU harness on TB2 ===")
    print(f"input harness : {CODE_AGENT_SIMPLE}")
    print(f"actor model   : {args.model} (env={args.env}, k={args.k}, timeout x{args.timeout_multiplier})")
    print(f"tasks ({len(args.tasks)}): {args.tasks}")
    print(f"  final_exam (locked): {split.final_exam}")
    print(f"  audit             : {split.audit}")
    print(f"  judging (gate)    : {split.judging}")
    print(f"  practice          : {split.practice}")
    print(f"loop: rounds={cfg.loop.rounds} segment_length={cfg.loop.segment_length} "
          f"strategies/round={cfg.loop.strategies_per_round} wobble_runs={cfg.loop.wobble_runs}")
    demo_jobs = Path(args.workspace) / "_demo_jobs"
    demo_ds = Path(args.workspace) / "_demo_ds"
    sample_eval = split.judging or args.tasks[:1]
    print("\nexact harbor command the gate will run (per eval):")
    print("  " + " ".join(bench.build_cmd(src, sample_eval, demo_jobs, demo_ds)))

    # Rough cost envelope so the spend is never a surprise.
    judging_evals_per_round = 2 * len(split.judging)  # old + new harness
    print(f"\napprox task-evals/round at the gate: ~{judging_evals_per_round} "
          f"(+ practice {cfg.piles.practice}, + {cfg.loop.strategies_per_round} agent proposals)")
    print(f"approx upper-bound task-evals: ~{cfg.loop.rounds * (judging_evals_per_round + cfg.piles.practice)} "
          f"(before caching). At ~10-45 min/eval this is the wall-clock driver.")

    if args.dry_run:
        print("\n[dry-run] no Docker, no model calls, no spend.")
        return

    from studio.orchestrator import Orchestrator

    print(f"\nworkspace: {args.workspace}\n--- running optimizer (this spends real compute) ---")
    orch = Orchestrator(
        workspace=args.workspace, source_harness=src, benchmark=bench,
        backend=backend, config=cfg, part_map=nexau_part_map(),
    )
    result = orch.run()
    best_dir = orch.state.root / "best"
    print("\n=== RESULT ===")
    print(f"baseline final-exam : {result.baseline_final:.3f}")
    print(f"optimized final-exam: {result.final_score:.3f}")
    print(f"uplift              : {result.uplift:+.3f}")
    print(f"wobble (noise floor): {result.wobble:.3f}")
    print(f"accepted edits      : {result.accepted}")
    print(f"task_runs (cost)    : {result.task_runs}  cache_hits={result.cache_hits}")
    print(f"cost_per_point      : {result.cost_per_point:.1f} task-runs / pp uplift")
    print(f"best harness        : {best_dir}")

    import json
    out = Path(args.workspace) / "tb2_ours.json"
    summary = {
        "label": "ours",
        "pass_rate": round(result.final_score, 4),
        "baseline": round(result.baseline_final, 4),
        "uplift": round(result.uplift, 4),
        "accepted": result.accepted,
        "task_runs": result.task_runs,
        "cost_per_point": (None if result.cost_per_point == float("inf") else round(result.cost_per_point, 2)),
        "best_dir": str(best_dir),
        "n_final_tasks": len(split.final_exam),
    }
    out.write_text(json.dumps(summary, indent=2))
    Path("/tmp/tb2_ours.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out} (and /tmp/tb2_ours.json)")


if __name__ == "__main__":
    main()
