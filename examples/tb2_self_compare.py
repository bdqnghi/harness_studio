#!/usr/bin/env python
"""Per-backbone SELF-HARNESS self-comparison on TB2 (actor = proposer = backbone).

One *cell* of the {nexau, mini-swe} x {gemini, gpt-5.4} matrix. The backbone
improves ITS OWN harness: it proposes the edits (proposer) and drives the tasks
(actor). The flow:

  0. Freeze one split from metadata + a declared noise prior:
     held-in pool + regression + locked test.
  1. Measure the baseline on held-in tasks only OR validate supplied held-in
     rates (``--baseline-score`` / ``--baseline-json``) to skip that run.
  2. Run the SHO optimizer on the held-in pool, gated by pool + regression.
  3. Verdict: score baseline vs optimized on the LOCKED test at test_k -> a
     per-backbone lift with an error bar.

  # one cell (dry-run: no Docker, no spend — prints the plan)
  python examples/tb2_self_compare.py --harness nexau --backbone gemini --dry-run

  # the full 2x2 matrix, gemini-key cells and openai-key cells in parallel
  python examples/tb2_self_compare.py --matrix
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.backends.factory import make_backend  # noqa: E402
from studio.components.calibration import (  # noqa: E402
    DEFAULT_TASK_CACHE, Calibration, TaskStat, calibrate, compute_sigma2,
    read_difficulty_meta, read_task_timeouts,
)
from studio.components.splitter import choose_split, detectable_delta  # noqa: E402
from studio.config import (  # noqa: E402
    Config, EditConfig, GateConfig, LoopConfig, PileConfig,
)
from studio.harness import Harness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AHE_DIR = Path(
    os.environ.get("AHE_DIR", str(REPO_ROOT.parent / "agentic-harness-engineering"))
)
# Curated mini-swe-agent harness (installable subset: pyproject.toml + src/).
MINISWE_HARNESS = REPO_ROOT / "artifacts" / "mini_swe_harness"

# A canonical backbone -> the model string each role/target needs. nexau takes a
# bare model name; mini-swe + the proposer use litellm format.
@dataclass(frozen=True)
class BackboneSpec:
    provider: str
    model: str

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        model = self.model.strip()
        if not provider or not model:
            raise ValueError("provider and model must be non-empty")
        if "/" in model and not model.startswith(f"{provider}/"):
            raise ValueError(
                f"model prefix does not match provider: {provider!r} vs {model!r}"
            )
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)

    @property
    def litellm_model(self) -> str:
        """Provider-qualified model name required by mini-swe and accepted by LiteLLM."""
        prefix = f"{self.provider}/"
        return self.model if self.model.startswith(prefix) else prefix + self.model

    @property
    def nexau_model(self) -> str:
        """NexAU selects the provider separately through LLM_API_TYPE."""
        prefix = f"{self.provider}/"
        return self.model[len(prefix):] if self.model.startswith(prefix) else self.model


BACKBONES = {
    "gemini": BackboneSpec("gemini", "gemini-3.5-flash"),
    "gpt-5.4": BackboneSpec("openai", "gpt-5.4"),
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
    return list(dict.fromkeys(names))


def resolve_backbone(args) -> BackboneSpec:
    if args.provider or args.model:
        if not args.provider or not args.model:
            raise ValueError("--provider and --model must be supplied together")
        return BackboneSpec(args.provider, args.model)
    return BACKBONES[args.backbone]


def make_target(harness: str, spec: BackboneSpec, *, real: bool, k: int,
                n_concurrent: int, timeout_multiplier: float, ahe_dir: Path):
    """Return (benchmark, source_harness, part_map, proposer_model)."""
    if harness == "nexau":
        from studio.benchmark.nexau import NexauBenchmark
        from examples.run_nexau_tb2 import nexau_part_map
        bench = NexauBenchmark(real=real, ahe_dir=ahe_dir, model=spec.nexau_model,
                               provider=spec.provider,
                               k=k, n_concurrent=n_concurrent, timeout_multiplier=timeout_multiplier)
        return bench, Harness(ahe_dir / "agents" / "code_agent_simple"), nexau_part_map(), spec.litellm_model
    from studio.benchmark.mini_swe import MiniSweBenchmark, mini_swe_part_map
    bench = MiniSweBenchmark(real=real, ahe_dir=ahe_dir, model=spec.litellm_model,
                             k=k, n_concurrent=n_concurrent, timeout_multiplier=timeout_multiplier)
    return bench, Harness(MINISWE_HARNESS), mini_swe_part_map(), spec.litellm_model


def _cell_config(split, *, seed, args) -> Config:
    return Config(
        seed=seed,
        piles=PileConfig(practice=min(args.round_size, max(1, len(split.practice))),
                         judging=len(split.judging), audit=len(split.audit),
                         final_exam=len(split.final_exam)),
        loop=LoopConfig(rounds=args.rounds, segment_length=args.segment_length,
                        wobble_runs=args.wobble_runs, strategies_per_round=args.strategies),
        gate=GateConfig(borderline_extra_runs=args.borderline_runs),
        edits=EditConfig(budget_per_part=args.budget),
    )


def _provided_baseline(args, calibration_tasks, timeouts) -> Calibration:
    """Build a Calibration from a user-supplied baseline number (skip the run).

    Only the already-frozen held-in calibration tasks are consumed. Extra JSON
    entries (including future locked tasks) are intentionally ignored.
    """
    if args.baseline_json:
        data = json.loads(Path(args.baseline_json).read_text())
        if not isinstance(data, dict):
            raise ValueError("--baseline-json must contain an object {task: rate}")
        missing = [t for t in calibration_tasks if t not in data]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                "--baseline-json is missing held-in calibration tasks: "
                f"{preview}{suffix}"
            )
        p_by = {t: float(data[t]) for t in calibration_tasks}
    else:
        if args.baseline_sigma2 is None:
            raise ValueError(
                "--baseline-score is an aggregate and cannot estimate per-task "
                "stochasticity; supply --baseline-sigma2"
            )
        p_by = {t: float(args.baseline_score) for t in calibration_tasks}
    if any(not 0.0 <= p <= 1.0 for p in p_by.values()):
        raise ValueError("provided baseline rates must be in [0, 1]")
    if args.baseline_sigma2 is not None:
        if not 0.01 <= args.baseline_sigma2 <= 0.25:
            raise ValueError("--baseline-sigma2 must be in [0.01, 0.25]")
        sigma2 = float(args.baseline_sigma2)
    else:
        if p_by and all(p in (0.0, 1.0) for p in p_by.values()):
            raise ValueError(
                "binary-only provided baselines do not estimate stochastic noise; "
                "supply --baseline-sigma2 or rates measured over repeated rollouts"
            )
        sigma2 = compute_sigma2(p_by, k=args.calibration_k)
    stats = {t: TaskStat(p=p_by[t],
                         runtime_sec=float(timeouts.get(t, 600.0)), k=args.calibration_k)
             for t in calibration_tasks}
    return Calibration(stats=stats, sigma2=sigma2, model=resolve_backbone(args).litellm_model)


def run_cell(args) -> dict:
    tasks = _opt_tasks(args)
    if not tasks:
        raise ValueError("no benchmark tasks were selected")
    spec = resolve_backbone(args)
    cell_name = (
        args.backbone
        if args.provider is None and args.model is None
        else spec.litellm_model
    )
    timeouts = read_task_timeouts(tasks, cache=Path(args.task_cache))
    diffs_meta = read_difficulty_meta(tasks, cache=Path(args.task_cache))
    # Gate/optimize at opt_k (cheap, robust via the dual-split gate); score the
    # VERDICT at test_k (trustworthy). The heavy TB2 tasks make all-k=3 brutal.
    opt_bench, src, part_map, proposer_model = make_target(
        args.harness, spec, real=not args.dry_run, k=args.opt_k,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
        ahe_dir=Path(args.ahe_dir))
    calibration_bench, _, _, _ = make_target(
        args.harness, spec, real=not args.dry_run, k=args.calibration_k,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
        ahe_dir=Path(args.ahe_dir))
    test_bench, _, _, _ = make_target(
        args.harness, spec, real=not args.dry_run, k=args.test_k,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
        ahe_dir=Path(args.ahe_dir))

    print(f"=== self-harness cell: harness={args.harness} backbone={cell_name} ===")
    print(f"actor=proposer model: {proposer_model} | tasks N={len(tasks)} | "
          f"calibration_k={args.calibration_k} opt_k={args.opt_k} "
          f"test_k={args.test_k} round_size={args.round_size}")

    # --- step 0: choose the split without touching future locked tasks ---
    provided = bool(args.baseline_json) or args.baseline_score is not None
    sigma2 = args.sigma2_prior
    if args.dry_run:
        print(f"planning: dry-run (sigma2 prior={sigma2})")
    elif provided:
        print(
            f"planning: sigma2 prior={sigma2:.3f}; supplied baseline will be "
            "restricted to held-in tasks after the split is frozen"
        )
    else:
        # Use a conservative prior to freeze the split. Measured calibration
        # happens only after locked tasks have been removed.
        sigma2 = args.sigma2_prior
        print(f"planning: sigma2 prior={sigma2:.3f}; locked tasks not evaluated")

    # --- step 1: choose the single split ---
    plan = choose_split(
        tasks, sigma2=sigma2, round_size=args.round_size, difficulties=diffs_meta, timeouts=timeouts,
        seed=args.seed, opt_k=args.opt_k, test_k=args.test_k, delta_round=args.delta_round,
        reg_cap=args.reg_cap, test_floor=args.test_floor,
        test_budget_cap=args.test_budget_cap, heavy_sec=args.heavy_sec)
    split = plan.split

    if args.dry_run:
        hv_gate = sum(
            1 for t in (set(split.judging) | set(split.regression) | set(split.audit))
            if timeouts.get(t, 0.0) >= args.heavy_sec
        )
        print(f"plan [{plan.mode}]: {plan.rationale}")
        print(f"  pool={len(split.practice)} judging={len(split.judging)} "
              f"regression={len(split.regression)} audit={len(split.audit)} "
              f"test={len(split.final_exam)} | heavy-in-gate={hv_gate}")
        print("[dry-run] no Docker, no model calls, no spend.")
        return {"cell": f"{args.harness}:{cell_name}", "mode": plan.mode, "dry_run": True}

    if plan.mode == "transfer":
        raise RuntimeError(
            "benchmark is too small for a locked holdout and no transfer benchmark "
            "is configured; refusing to optimize or emit a verdict"
        )

    calibration_tasks = list(dict.fromkeys(split.practice + split.regression))
    if provided:
        cal = _provided_baseline(args, calibration_tasks, timeouts)
        sigma2 = cal.sigma2
        print(
            f"baseline: accepted supplied rates for {len(calibration_tasks)} "
            f"held-in tasks only; sigma2={sigma2:.3f}"
        )
    else:
        cal = calibrate(
            calibration_bench, src, calibration_tasks, k=args.calibration_k,
            runtimes=timeouts, model=spec.litellm_model,
        )
        sigma2 = cal.sigma2
        print(
            f"baseline: calibrated {len(calibration_tasks)} held-in tasks only; "
            f"sigma2={sigma2:.3f}, mean p="
            f"{statistics.mean(cal.difficulties().values()):.3f}"
        )
    plan = replace(
        plan,
        sigma2=sigma2,
        detectable_round=detectable_delta(
            len(split.judging), sigma2, k=args.opt_k
        ),
        detectable_final=detectable_delta(
            len(split.final_exam), sigma2, k=args.test_k
        ),
    )

    hv_gate = sum(
        1 for t in (set(split.judging) | set(split.regression) | set(split.audit))
        if timeouts.get(t, 0.0) >= args.heavy_sec
    )
    print(f"plan [{plan.mode}]: {plan.rationale}")
    print(f"  pool={len(split.practice)} judging={len(split.judging)} "
          f"regression={len(split.regression)} audit={len(split.audit)} "
          f"test={len(split.final_exam)} | heavy-in-gate={hv_gate} | "
          f"calibrated detectable round={plan.detectable_round:.3f} "
          f"test={plan.detectable_final:.3f}")

    from studio.orchestrator import Orchestrator

    # --- step 2: optimize on the held-in pool (gated by pool + regression) ---
    ws = Path(args.workspace)
    # The optimizer never receives locked task ids. Only the external verdict
    # below can access them.
    optimization_split = replace(split, final_exam=[])
    cfg = _cell_config(optimization_split, seed=args.seed, args=args)
    orch = Orchestrator(workspace=ws, source_harness=src, benchmark=opt_bench,
                        backend=make_backend(proposer_model, log_dir=ws / "proposer-logs"),
                        config=cfg, part_map=part_map, split=optimization_split)
    orch.run()
    optimized = orch.harness

    # --- step 3: verdict on the LOCKED test at test_k ---
    base = test_bench.run(src, split.final_exam, run_idx=0)
    opt = test_bench.run(optimized, split.final_exam, run_idx=0)
    per_task_lift = {t: opt.get(t, 0.0) - base.get(t, 0.0) for t in split.final_exam}

    lift = statistics.mean(per_task_lift.values()) if per_task_lift else 0.0
    stdev = statistics.stdev(per_task_lift.values()) if len(per_task_lift) > 1 else 0.0
    se = stdev / (len(per_task_lift) ** 0.5) if per_task_lift else 0.0
    result = {
        "cell": f"{args.harness}:{cell_name}", "mode": plan.mode, "n_tasks": len(per_task_lift),
        "lift": round(lift, 4), "se": round(se, 4), "sigma2": round(sigma2, 4),
        "detectable_final": round(plan.detectable_final, 4), "per_task_lift": per_task_lift,
    }
    out = ws / "lift.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n=== CELL VERDICT {result['cell']}: lift {lift:+.3f} ± {se:.3f} "
          f"(detectable {plan.detectable_final:.3f}) -> {out}")
    return result


def _matrix_child_cmd(args, *, harness: str, backbone: str, workspace: Path) -> list[str]:
    cmd = [
        sys.executable, __file__,
        "--harness", harness,
        "--backbone", backbone,
        "--workspace", str(workspace),
        "--task-cache", args.task_cache,
        "--ahe-dir", args.ahe_dir,
        "--calibration-k", str(args.calibration_k),
        "--opt-k", str(args.opt_k),
        "--test-k", str(args.test_k),
        "--round-size", str(args.round_size),
        "--rounds", str(args.rounds),
        "--segment-length", str(args.segment_length),
        "--strategies", str(args.strategies),
        "--wobble-runs", str(args.wobble_runs),
        "--borderline-runs", str(args.borderline_runs),
        "--budget", str(args.budget),
        "--n-concurrent", str(args.n_concurrent),
        "--timeout-multiplier", str(args.timeout_multiplier),
        "--seed", str(args.seed),
        "--sigma2-prior", str(args.sigma2_prior),
        "--delta-round", str(args.delta_round),
        "--reg-cap", str(args.reg_cap),
        "--test-floor", str(args.test_floor),
        "--test-budget-cap", str(args.test_budget_cap),
        "--heavy-sec", str(args.heavy_sec),
    ]
    if args.tasks:
        cmd += ["--tasks", args.tasks]
    if args.baseline_score is not None:
        cmd += ["--baseline-score", str(args.baseline_score)]
    if args.baseline_json:
        cmd += ["--baseline-json", args.baseline_json]
    if args.baseline_sigma2 is not None:
        cmd += ["--baseline-sigma2", str(args.baseline_sigma2)]
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_matrix(args) -> None:
    """Run the 4 cells in quota-clean WAVES: one harness at a time, its two
    backbones in parallel (gemini-key + openai-key -> separate quotas, no 429
    contention). gemini never runs two cells at once."""
    if args.provider or args.model:
        raise ValueError("--provider/--model configure one cell and cannot be combined with --matrix")
    Path(args.workspace).mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for harness in HARNESSES:
        procs = []
        for backbone in BACKBONES:
            ws = Path(args.workspace) / f"{harness}_{backbone}".replace("/", "-")
            cmd = _matrix_child_cmd(
                args, harness=harness, backbone=backbone, workspace=ws
            )
            log = Path(args.workspace) / f"{harness}_{backbone}.log".replace("/", "-")
            log_handle = log.open("w")
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            procs.append((f"{harness}:{backbone}", proc, log, log_handle))
            print(f"[wave {harness}] launched {harness}:{backbone} -> {log}", flush=True)
        for name, p, log, log_handle in procs:
            p.wait()
            log_handle.close()
            print(f"[wave {harness}] {name} exited rc={p.returncode}", flush=True)
            if p.returncode:
                failures.append(f"{name} (rc={p.returncode}, log={log})")
    # Collect the 2x2 lift table.
    print("\n=== 2x2 LIFT TABLE ===")
    for harness in HARNESSES:
        row = []
        for backbone in BACKBONES:
            f = Path(args.workspace) / f"{harness}_{backbone}".replace("/", "-") / "lift.json"
            if f.exists():
                d = json.loads(f.read_text())
                row.append(f"{backbone}={d['lift']:+.3f}±{d['se']:.3f}")
            elif args.dry_run:
                row.append(f"{backbone}=dry-run")
            else:
                row.append(f"{backbone}=?")
        print(f"  {harness:9s} | " + " | ".join(row))
    if failures:
        raise RuntimeError("matrix cells failed: " + "; ".join(failures))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=HARNESSES, default="nexau")
    ap.add_argument("--backbone", choices=list(BACKBONES), default="gemini")
    ap.add_argument("--provider", default=None,
                    help="provider for a custom single-cell model (use with --model)")
    ap.add_argument("--model", default=None,
                    help="custom single-cell model name (use with --provider)")
    ap.add_argument("--matrix", action="store_true", help="run the full 2x2 in parallel")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma-sep task ids (default: all cached)")
    ap.add_argument("--task-cache", default=str(DEFAULT_TASK_CACHE))
    ap.add_argument("--ahe-dir", default=str(DEFAULT_AHE_DIR))
    ap.add_argument("--workspace", default="/tmp/sho_self")
    ap.add_argument("--calibration-k", type=int, default=3,
                    help="rollouts/task used to estimate baseline stochasticity")
    ap.add_argument("--opt-k", type=int, default=1, help="rollouts/task during the gate (cheap)")
    ap.add_argument("--test-k", type=int, default=3, help="rollouts/task for the final verdict (trustworthy)")
    ap.add_argument("--round-size", type=int, default=32, help="tasks run per round (the mini-batch)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--segment-length", type=int, default=2)
    ap.add_argument("--strategies", type=int, default=2)
    ap.add_argument("--wobble-runs", type=int, default=3)
    ap.add_argument("--borderline-runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--n-concurrent", type=int, default=12)
    ap.add_argument("--timeout-multiplier", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma2-prior", type=float, default=0.2, help="sigma2 for dry-run preview")
    ap.add_argument("--delta-round", type=float, default=0.12)
    ap.add_argument("--reg-cap", type=int, default=32)
    ap.add_argument("--test-floor", type=int, default=25)
    ap.add_argument("--test-budget-cap", type=int, default=0, help=">0: grade a representative test subsample")
    ap.add_argument("--heavy-sec", type=float, default=3600.0)
    ap.add_argument("--baseline-score", type=float, default=None,
                    help="provide one aggregate held-in pass-rate; requires --baseline-sigma2")
    ap.add_argument("--baseline-json", default=None,
                    help="provide every held-in task's pass-rate as JSON {task: rate}")
    ap.add_argument("--baseline-sigma2", type=float, default=None,
                    help="noise variance for an aggregate or binary-only provided baseline")
    args = ap.parse_args()
    if args.calibration_k < 3:
        ap.error("--calibration-k must be at least 3 to estimate stochasticity")
    if args.wobble_runs < 3:
        ap.error("--wobble-runs must be at least 3")
    if args.opt_k < 1:
        ap.error("--opt-k must be positive")
    if args.test_k < 3:
        ap.error("--test-k must be at least 3 for a noise-honest verdict")
    if not 0.01 <= args.sigma2_prior <= 0.25:
        ap.error("--sigma2-prior must be in [0.01, 0.25]")
    if args.baseline_json and args.baseline_score is not None:
        ap.error("use only one of --baseline-json and --baseline-score")
    if args.baseline_score is not None and args.baseline_sigma2 is None:
        ap.error("--baseline-score requires --baseline-sigma2")
    if args.matrix:
        run_matrix(args)
    else:
        run_cell(args)


if __name__ == "__main__":
    main()
