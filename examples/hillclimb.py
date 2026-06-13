#!/usr/bin/env python
"""Target-agnostic hill-climb driver — run SHO on ANY registered benchmark.

This is the generalized replacement for the TB2-specific driver: it resolves a
``Target`` from the registry, handles warm-start (mutate the shipped baseline
harness) OR cold-start (synthesize a harness when none ships), runs the
optimizer, and reports the verdict against the target's published baseline.

  # preview the plan (no spend)
  python examples/hillclimb.py --target tau2-telecom --dry-run

  # warm-start hill-climb on tau2 telecom (beat published Pass^1 0.34)
  python examples/hillclimb.py --target tau2-telecom --model gpt-4.1 \
      --user-model gpt-4.1-mini --rounds 8 --optimizer tree

  # cold-start: synthesize a harness from nothing, then climb
  python examples/hillclimb.py --target browsecomp --cold-start --model gpt-4.1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import studio.targets_builtin  # noqa: E402,F401  (populates the Target registry)
from studio.backends.factory import make_backend  # noqa: E402
from studio.components.splitter import TaskSplit, detectable_delta  # noqa: E402
from studio.config import Config, EditConfig, GateConfig, LoopConfig, PileConfig  # noqa: E402
from studio.targets import TargetConfig, get_target, list_targets  # noqa: E402


def simple_split(tasks: list[str], *, seed: int, held_out: int, reg: int,
                 judging: int, audit: int) -> TaskSplit:
    """Deterministic split for fast benchmarks (no heavy-task handling needed).

    final_exam (locked) | regression (disjoint do-no-harm) | practice pool
    (judging + audit are slices of the pool)."""
    import hashlib

    order = sorted(tasks, key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
    held_out = min(held_out, max(0, len(order) - 4))
    final_exam = order[:held_out]
    rest = order[held_out:]
    regression = rest[:reg]
    pool = rest[reg:]
    if not pool:  # tiny task sets: borrow back from regression
        pool, regression = regression, []
    judging = pool[: min(judging, len(pool))] or pool[:1]
    audit = pool[: min(audit, len(pool))] or judging
    return TaskSplit(practice=pool, judging=judging, audit=audit,
                     final_exam=final_exam, regression=regression)


def run(args) -> dict:
    target = get_target(args.target)
    ws = Path(args.workspace)
    ws.mkdir(parents=True, exist_ok=True)
    proposer_model = args.proposer_model or args.model

    extra = {"user_model": args.user_model} if args.user_model else {}
    opt_cfg = TargetConfig(model=args.model, k=args.opt_k, n_concurrent=args.n_concurrent,
                           real=not args.dry_run, extra=extra)
    test_cfg = replace(opt_cfg, k=args.test_k)
    opt_bench = target.make_benchmark(opt_cfg)
    test_bench = target.make_benchmark(test_cfg)

    tasks = opt_bench.list_tasks()
    if not tasks:
        raise SystemExit(f"target {args.target} returned no tasks")
    # Scale the split to task count: locked test gets ~40% (capped at --held-out),
    # the rest is held-in (regression + pool); clamp slices to what's available.
    n = len(tasks)
    held_out = min(args.held_out, max(4, int(round(n * 0.4))))
    avail = n - held_out
    reg = min(args.reg, max(0, avail // 3))
    pool_n = avail - reg
    judging = min(args.judging, pool_n)
    audit = min(args.audit, pool_n)
    split = simple_split(tasks, seed=args.seed, held_out=held_out, reg=reg,
                         judging=judging, audit=audit)

    mode = "cold-start" if (args.cold_start or target.seed_harness() is None) else "warm-start"
    print(f"=== hillclimb: target={args.target} mode={mode} model={args.model} "
          f"optimizer={args.optimizer} localizer={args.localizer} ===")
    print(f"tasks N={len(tasks)} | held-in pool={len(split.practice)} judging={len(split.judging)} "
          f"regression={len(split.regression)} audit={len(split.audit)} | locked test={len(split.final_exam)}")
    print(f"baseline bar: {target.baseline_score} ({target.baseline_note})")
    det = detectable_delta(len(split.final_exam), args.sigma2, k=args.test_k) if split.final_exam else 0.0
    print(f"detectable on locked test @k={args.test_k}: ~{det:.3f}")
    if args.dry_run:
        print("[dry-run] no spend.")
        return {"target": args.target, "mode": mode, "dry_run": True,
                "n_tasks": len(tasks), "split": {"pool": len(split.practice),
                "test": len(split.final_exam)}, "baseline": target.baseline_score}

    backend = make_backend(proposer_model, log_dir=ws / "proposer-logs")

    # round-0 harness: warm (shipped seed) or cold (synthesized from the brief).
    seed = target.resolve_seed(backend, ws, force_cold=args.cold_start)
    ok, err = opt_bench.boot_check(seed)
    if not ok:
        raise SystemExit(f"seed harness failed boot_check: {err}")

    from studio.orchestrator import Orchestrator

    cfg = Config(
        seed=args.seed,
        score_cache=str(ws / "score_cache.jsonl"),
        piles=PileConfig(practice=min(args.round_size, len(split.practice)),
                         judging=len(split.judging), audit=len(split.audit),
                         final_exam=0),
        loop=LoopConfig(rounds=args.rounds, segment_length=args.segment_length,
                        wobble_runs=args.wobble_runs, strategies_per_round=args.strategies,
                        optimizer=args.optimizer, hypotheses_per_direction=args.hypotheses,
                        localizer=args.localizer),
        gate=GateConfig(borderline_extra_runs=args.borderline_runs,
                        aggregate_accept=args.aggregate_accept),
        edits=EditConfig(budget_per_part=args.budget),
    )
    optimization_split = replace(split, final_exam=[])  # optimizer never sees locked test
    orch = Orchestrator(workspace=ws, source_harness=seed, benchmark=opt_bench,
                        backend=backend, config=cfg, part_map=target.part_map(),
                        split=optimization_split)
    orch.run()
    optimized = orch.harness

    # verdict on the locked test at test_k: seed vs optimized, paired per-task lift.
    base = test_bench.run(seed, split.final_exam, run_idx=0)
    opt = test_bench.run(optimized, split.final_exam, run_idx=0)
    per_task = {t: opt.get(t, 0.0) - base.get(t, 0.0) for t in split.final_exam}
    base_mean = statistics.mean(base.values()) if base else 0.0
    opt_mean = statistics.mean(opt.values()) if opt else 0.0
    lift = statistics.mean(per_task.values()) if per_task else 0.0
    se = (statistics.stdev(per_task.values()) / len(per_task) ** 0.5) if len(per_task) > 1 else 0.0
    result = {
        "target": args.target, "mode": mode, "model": args.model,
        "n_test": len(split.final_exam),
        "baseline_harness_score": round(base_mean, 4),
        "optimized_harness_score": round(opt_mean, 4),
        "lift": round(lift, 4), "se": round(se, 4),
        "published_baseline": target.baseline_score,
        "detectable": round(det, 4),
        "per_task_lift": per_task,
    }
    (ws / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n=== VERDICT {args.target} [{mode}] ===")
    print(f"baseline harness: {base_mean:.3f}  ->  SHO-optimized: {opt_mean:.3f}  "
          f"(lift {lift:+.3f} ± {se:.3f}, detectable {det:.3f})")
    print(f"vs published baseline {target.baseline_score}: "
          f"{'BEATS' if opt_mean > (target.baseline_score or 0) else 'below'} the published bar")
    print(f"-> {ws / 'result.json'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help=f"one of: {list_targets()}")
    ap.add_argument("--model", default="gpt-4.1", help="agent + proposer model (litellm)")
    ap.add_argument("--proposer-model", default=None, help="override proposer model (default: --model)")
    ap.add_argument("--user-model", default=None, help="tau2 user-simulator model")
    ap.add_argument("--cold-start", action="store_true", help="synthesize a harness even if a seed exists")
    ap.add_argument("--optimizer", choices=("classic", "tree"), default="tree")
    ap.add_argument("--localizer", choices=("off", "inline", "agentic", "auto"),
                    default="auto", help="evidence-grounded context localization "
                    "(off=legacy diagnosis-only); applies to both optimizer paths")
    ap.add_argument("--aggregate-accept", action="store_true",
                    help="gate accepts on POOLED held-in gain (judging∪regression) "
                         "instead of per-slice do-no-harm — for noisy benchmarks where "
                         "gains land on one slice")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workspace", default="/tmp/sho_hillclimb")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--segment-length", type=int, default=2)
    ap.add_argument("--strategies", type=int, default=2)
    ap.add_argument("--hypotheses", type=int, default=4)
    ap.add_argument("--round-size", type=int, default=16)
    ap.add_argument("--wobble-runs", type=int, default=3)
    ap.add_argument("--borderline-runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--opt-k", type=int, default=1)
    ap.add_argument("--test-k", type=int, default=3)
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma2", type=float, default=0.2, help="noise prior for the detectable-delta report")
    ap.add_argument("--held-out", type=int, default=40, help="locked test size")
    ap.add_argument("--reg", type=int, default=16, help="regression (do-no-harm) set size")
    ap.add_argument("--judging", type=int, default=16, help="stable gate slice of the pool")
    ap.add_argument("--audit", type=int, default=8, help="deep-audit slice of the pool")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
