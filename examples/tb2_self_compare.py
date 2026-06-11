#!/usr/bin/env python
"""Per-backbone SELF-HARNESS self-comparison on TB2 (actor = proposer = backbone).

One *cell* of the {nexau, mini-swe} x {gemini, gpt-5.4} matrix. The backbone
improves ITS OWN harness: it proposes the edits (proposer) and drives the tasks
(actor). We calibrate the baseline, choose a power-based split (CV for TB2's
N=89), run the SHO optimizer per fold, then score baseline vs optimized on each
fold's locked test slice at k -> a per-backbone lift with an error bar.

  # one cell (dry-run: no Docker, no spend — prints the plan)
  python examples/tb2_self_compare.py --harness nexau --backbone gemini --dry-run

  # the full 2x2 matrix, gemini-key cells and openai-key cells in parallel
  python examples/tb2_self_compare.py --matrix
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.components.calibration import (  # noqa: E402
    DEFAULT_TASK_CACHE, calibrate, read_difficulty_meta, read_task_timeouts,
)
from studio.components.splitter import choose_eval_plan  # noqa: E402
from studio.config import (  # noqa: E402
    Config, EditConfig, GateConfig, LoopConfig, PileConfig,
)
from studio.harness import Harness  # noqa: E402

DEFAULT_AHE_DIR = Path("/home/nghibui/codes/agentic-harness-engineering")
CODE_AGENT_SIMPLE = DEFAULT_AHE_DIR / "agents" / "code_agent_simple"
# Curated mini-swe-agent harness (installable subset: pyproject.toml + src/).
MINISWE_HARNESS = Path("/home/nghibui/codes/harness_studio/artifacts/mini_swe_harness")

# A canonical backbone -> the model string each role/target needs. nexau takes a
# bare model name; mini-swe + the proposer use litellm format.
BACKBONES = {
    "gemini": {
        "nexau_actor": "gemini-3.5-flash",
        "miniswe_actor": "gemini/gemini-3.5-flash",
        "proposer": "gemini/gemini-3.5-flash",
    },
    "gpt-5.4": {
        "nexau_actor": "gpt-5.4",
        "miniswe_actor": "gpt-5.4",
        "proposer": "gpt-5.4",
    },
}
HARNESSES = ("nexau", "mini-swe")


def _opt_tasks(args) -> list[str]:
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",") if t.strip()]
    # default: every cached TB2 task (the full N=89 set)
    names: list[str] = []
    for hash_dir in sorted(Path(args.task_cache).iterdir()):
        if hash_dir.is_dir():
            for td in sorted(hash_dir.iterdir()):
                if td.is_dir() and (td / "task.toml").exists():
                    names.append(td.name)
    return names


def make_target(harness: str, backbone: str, *, real: bool, k: int,
                n_concurrent: int, timeout_multiplier: float, ahe_dir: Path):
    """Return (benchmark, source_harness, part_map, proposer_model)."""
    bb = BACKBONES[backbone]
    if harness == "nexau":
        from studio.benchmark.nexau import NexauBenchmark
        from examples.run_nexau_tb2 import nexau_part_map
        bench = NexauBenchmark(real=real, ahe_dir=ahe_dir, model=bb["nexau_actor"],
                               k=k, n_concurrent=n_concurrent, timeout_multiplier=timeout_multiplier)
        return bench, Harness(CODE_AGENT_SIMPLE), nexau_part_map(), bb["proposer"]
    from studio.benchmark.mini_swe import MiniSweBenchmark, mini_swe_part_map
    bench = MiniSweBenchmark(real=real, ahe_dir=ahe_dir, model=bb["miniswe_actor"],
                             k=k, n_concurrent=n_concurrent, timeout_multiplier=timeout_multiplier)
    return bench, Harness(MINISWE_HARNESS), mini_swe_part_map(), bb["proposer"]


def _fold_config(fold, *, seed, args) -> Config:
    return Config(
        seed=seed,
        piles=PileConfig(practice=min(args.practice_size, max(1, len(fold.practice))),
                         judging=len(fold.judging), audit=len(fold.audit),
                         final_exam=len(fold.final_exam)),
        loop=LoopConfig(rounds=args.rounds, segment_length=args.segment_length,
                        wobble_runs=args.wobble_runs, strategies_per_round=args.strategies),
        gate=GateConfig(borderline_extra_runs=args.borderline_runs),
        edits=EditConfig(budget_per_part=args.budget),
    )


def run_cell(args) -> dict:
    tasks = _opt_tasks(args)
    timeouts = read_task_timeouts(tasks, cache=Path(args.task_cache))
    diffs_meta = read_difficulty_meta(tasks, cache=Path(args.task_cache))
    # Gate/optimize at opt_k (cheap, robust via the dual-split gate); score the
    # VERDICT at test_k (trustworthy). The heavy TB2 tasks make all-k=3 brutal.
    opt_bench, src, part_map, proposer_model = make_target(
        args.harness, args.backbone, real=not args.dry_run, k=args.opt_k,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
        ahe_dir=Path(args.ahe_dir))
    test_bench, _, _, _ = make_target(
        args.harness, args.backbone, real=not args.dry_run, k=args.test_k,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
        ahe_dir=Path(args.ahe_dir))

    print(f"=== self-harness cell: harness={args.harness} backbone={args.backbone} ===")
    print(f"actor=proposer model: {proposer_model} | tasks N={len(tasks)} | opt_k={args.opt_k} test_k={args.test_k}")

    if args.dry_run:
        # No Docker: use a sigma2 prior + free metadata to preview the plan.
        plan = choose_eval_plan(tasks, sigma2=args.sigma2_prior, difficulties=diffs_meta,
                                timeouts=timeouts, seed=args.seed, k=args.test_k,
                                delta_step=args.delta_step, delta_final=args.delta_final,
                                val_budget_cap=args.val_budget_cap, heavy_sec=args.heavy_sec,
                                n_folds=args.n_folds)
        print(f"plan (sigma2 prior={args.sigma2_prior}): {plan.rationale}")
        folds = plan.folds if plan.mode == "kfold" else [plan.split]
        for i, f in enumerate(folds):
            hv = sum(1 for t in (set(f.judging) | set(f.gen) | set(f.audit)) if timeouts.get(t, 0) >= args.heavy_sec)
            print(f"  fold{i}: test={len(f.final_exam)} judging={len(f.judging)} gen={len(f.gen)} "
                  f"audit={len(f.audit)} practice={len(f.practice)} | heavy-in-gate={hv}")
        print("[dry-run] no Docker, no model calls, no spend.")
        return {"cell": f"{args.harness}:{args.backbone}", "mode": plan.mode, "dry_run": True}

    from studio.orchestrator import Orchestrator

    # 1. Calibrate the baseline once at opt_k (per-task difficulty + sigma2).
    cal = calibrate(opt_bench, src, tasks, k=args.opt_k, runtimes=timeouts, model=args.backbone)
    print(f"calibrated: sigma2={cal.sigma2:.3f} baseline mean p={statistics.mean(cal.difficulties().values()):.3f}")
    plan = choose_eval_plan(tasks, sigma2=cal.sigma2, difficulties=cal.difficulties(),
                            timeouts=timeouts, seed=args.seed, k=args.test_k,
                            delta_step=args.delta_step, delta_final=args.delta_final,
                            val_budget_cap=args.val_budget_cap, heavy_sec=args.heavy_sec,
                            n_folds=args.n_folds)
    print(f"plan: {plan.rationale}")
    folds = plan.folds if plan.mode == "kfold" else [plan.split]

    ws = Path(args.workspace)
    per_task_lift: dict[str, float] = {}
    for i, fold in enumerate(folds):
        cfg = _fold_config(fold, seed=args.seed, args=args)
        orch = Orchestrator(workspace=ws / f"fold_{i}", source_harness=src, benchmark=opt_bench,
                            backend=__import__("studio.backends.factory", fromlist=["make_backend"]).make_backend(
                                proposer_model, log_dir=ws / f"fold_{i}" / "proposer-logs"),
                            config=cfg, part_map=part_map, split=fold)
        orch.run()
        best = Harness(orch.state.root / "best")
        # Verdict: score baseline vs optimized on the LOCKED test slice at test_k.
        base = test_bench.run(src, fold.final_exam, run_idx=0)
        opt = test_bench.run(best, fold.final_exam, run_idx=0)
        for t in fold.final_exam:
            per_task_lift[t] = opt.get(t, 0.0) - base.get(t, 0.0)
        print(f"  fold{i}: test={len(fold.final_exam)} mean lift {statistics.mean([per_task_lift[t] for t in fold.final_exam]):+.3f}")

    lift = statistics.mean(per_task_lift.values()) if per_task_lift else 0.0
    stdev = statistics.pstdev(per_task_lift.values()) if len(per_task_lift) > 1 else 0.0
    se = stdev / (len(per_task_lift) ** 0.5) if per_task_lift else 0.0
    result = {
        "cell": f"{args.harness}:{args.backbone}", "mode": plan.mode, "n_tasks": len(per_task_lift),
        "lift": round(lift, 4), "se": round(se, 4), "sigma2": round(cal.sigma2, 4),
        "detectable_final": round(plan.detectable_final, 4), "per_task_lift": per_task_lift,
    }
    out = ws / "lift.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\n=== CELL VERDICT {result['cell']}: lift {lift:+.3f} ± {se:.3f} "
          f"(detectable {plan.detectable_final:.3f}) -> {out}")
    return result


def run_matrix(args) -> None:
    """Fire the 4 cells as parallel subprocesses (gemini-key pair + openai-key pair)."""
    procs = []
    for harness in HARNESSES:
        for backbone in BACKBONES:
            ws = Path(args.workspace) / f"{harness}_{backbone}".replace("/", "-")
            cmd = [sys.executable, __file__, "--harness", harness, "--backbone", backbone,
                   "--workspace", str(ws), "--task-cache", args.task_cache,
                   "--opt-k", str(args.opt_k), "--test-k", str(args.test_k),
                   "--rounds", str(args.rounds), "--n-concurrent", str(args.n_concurrent),
                   "--borderline-runs", str(args.borderline_runs)]
            if args.dry_run:
                cmd.append("--dry-run")
            log = Path(args.workspace) / f"{harness}_{backbone}.log".replace("/", "-")
            log.parent.mkdir(parents=True, exist_ok=True)
            procs.append((f"{harness}:{backbone}", subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT), log))
            print(f"launched {harness}:{backbone} -> {log}")
    for name, p, log in procs:
        p.wait()
        print(f"{name} exited rc={p.returncode}")
    # Collect the 2x2 lift table.
    print("\n=== 2x2 LIFT TABLE ===")
    for harness in HARNESSES:
        row = []
        for backbone in BACKBONES:
            f = Path(args.workspace) / f"{harness}_{backbone}".replace("/", "-") / "lift.json"
            if f.exists():
                d = json.loads(f.read_text())
                row.append(f"{backbone}={d['lift']:+.3f}±{d['se']:.3f}")
            else:
                row.append(f"{backbone}=?")
        print(f"  {harness:9s} | " + " | ".join(row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=HARNESSES, default="nexau")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="gemini")
    ap.add_argument("--matrix", action="store_true", help="run the full 2x2 in parallel")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma-sep task ids (default: all cached)")
    ap.add_argument("--task-cache", default=str(DEFAULT_TASK_CACHE))
    ap.add_argument("--ahe-dir", default=str(DEFAULT_AHE_DIR))
    ap.add_argument("--workspace", default="/tmp/sho_self")
    ap.add_argument("--opt-k", type=int, default=1, help="rollouts/task during calibration + gate (cheap)")
    ap.add_argument("--test-k", type=int, default=3, help="rollouts/task for the final verdict (trustworthy)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--segment-length", type=int, default=2)
    ap.add_argument("--strategies", type=int, default=2)
    ap.add_argument("--wobble-runs", type=int, default=1)
    ap.add_argument("--borderline-runs", type=int, default=1)
    ap.add_argument("--practice-size", type=int, default=6)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--n-concurrent", type=int, default=12)
    ap.add_argument("--timeout-multiplier", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma2-prior", type=float, default=0.2)
    ap.add_argument("--delta-step", type=float, default=0.12)
    ap.add_argument("--delta-final", type=float, default=0.05)
    ap.add_argument("--val-budget-cap", type=int, default=16)
    ap.add_argument("--heavy-sec", type=float, default=3600.0)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()
    if args.matrix:
        run_matrix(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
