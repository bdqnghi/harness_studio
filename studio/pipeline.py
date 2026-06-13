"""The SHO pipeline: resolve harness → profile → split → optimize → verdict.

Each stage is a function so the flow is explicit and unit-testable; the CLI
(``examples/hillclimb.py``) is a thin wrapper over ``run_hillclimb`` /
``profile_only``. The split is **difficulty-stratified** from a profile of the
input harness (warm shipped policy OR the cold-generated seed) unless
``--no-profile`` is set, in which case it falls back to a blind random split.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

from .backends.factory import make_backend
from .benchmark.instrument import InstrumentedBenchmark
from .components.profiler import Profile, profile_harness
from .components.splitter import (
    TaskSplit, _ordering, detectable_delta, random_split, stratified_split,
)
from .config import Config, EditConfig, GateConfig, LoopConfig, PileConfig
from .targets import TargetConfig, get_target


# --- stages ---------------------------------------------------------------

def resolve_harness(target, backend, ws: Path, *, cold: bool, validate):
    """Round-0 harness: warm (shipped seed) or cold (the coding agent generates
    one from the brief, retrying until it boots)."""
    return target.resolve_seed(backend, ws, force_cold=cold, validate=validate)


def profile(bench, harness, tasks, *, k: int, save_to: Path | None = None) -> Profile:
    """Run the harness over all tasks once → per-task pass/fail + trajectories."""
    def _note(batch, exc):
        print(f"[profile] a batch of {len(batch)} tasks failed ({type(exc).__name__}); skipped")
    prof = profile_harness(bench, harness, tasks, k=k, on_error=_note)
    if save_to is not None:
        prof.save(save_to)
    return prof


def build_split(args, *, profile: Profile | None, tasks: list[str]) -> TaskSplit:
    """Difficulty-stratified split from a profile, or a blind random split."""
    if profile is None:
        return random_split(tasks, seed=args.seed, held_in=args.held_in,
                            reg=args.reg, held_out_cap=args.held_out)
    return stratified_split(profile, held_in=args.held_in, reg=args.reg,
                            held_out_cap=args.held_out, seed=args.seed)


def verdict(test_bench, seed, optimized, held_out, *, k: int, sigma2: float) -> dict:
    """Grade seed vs optimized on the locked held_out → paired per-task lift ± SE."""
    base = test_bench.run(seed, held_out, run_idx=0)
    opt = test_bench.run(optimized, held_out, run_idx=0)
    per_task = {t: opt.get(t, 0.0) - base.get(t, 0.0) for t in held_out}
    base_mean = statistics.mean(base.values()) if base else 0.0
    opt_mean = statistics.mean(opt.values()) if opt else 0.0
    lift = statistics.mean(per_task.values()) if per_task else 0.0
    se = (statistics.stdev(per_task.values()) / len(per_task) ** 0.5) if len(per_task) > 1 else 0.0
    det = detectable_delta(len(held_out), sigma2, k=k) if held_out else 0.0
    return {"n_test": len(held_out), "baseline_harness_score": round(base_mean, 4),
            "optimized_harness_score": round(opt_mean, 4), "lift": round(lift, 4),
            "se": round(se, 4), "detectable": round(det, 4), "per_task_lift": per_task}


def _config(args, ws: Path, split: TaskSplit) -> Config:
    return Config(
        seed=args.seed,
        score_cache=str(ws / "score_cache.jsonl"),
        piles=PileConfig(round_size=min(args.round_size, max(1, len(split.held_in))),
                         regression=0, held_out=0),
        loop=LoopConfig(rounds=args.rounds, segment_length=args.segment_length,
                        wobble_runs=args.wobble_runs,
                        hypotheses_per_direction=args.hypotheses, localizer=args.localizer),
        gate=GateConfig(borderline_extra_runs=args.borderline_runs, strict_dual=args.strict_gate),
        edits=EditConfig(budget_per_part=args.budget),
    )


# --- composed flows -------------------------------------------------------

def _setup(args):
    target = get_target(args.target)
    ws = Path(args.workspace); ws.mkdir(parents=True, exist_ok=True)
    proposer_model = args.proposer_model or args.model
    extra = {"user_model": args.user_model} if args.user_model else {}
    opt_cfg = TargetConfig(model=args.model, k=args.opt_k, n_concurrent=args.n_concurrent,
                           real=not args.dry_run, extra=extra)
    opt_bench = target.make_benchmark(opt_cfg)
    tasks = opt_bench.list_tasks()
    if not tasks:
        raise SystemExit(f"target {args.target} returned no tasks")
    cap = getattr(args, "max_tasks", 0) or 0
    if cap and len(tasks) > cap:
        # Huge domains (e.g. tau2 telecom = 2285 tasks) are unprofileable whole;
        # take a deterministic seeded sample so the profile/split is reproducible.
        full = len(tasks)
        tasks = _ordering(tasks, args.seed)[:cap]
        print(f"[setup] {args.target} has {full} tasks; sampling {cap} (seed={args.seed})")
    mode = "cold-start" if (args.cold_start or target.seed_harness() is None) else "warm-start"
    return target, ws, proposer_model, opt_cfg, opt_bench, tasks, mode


def profile_only(args) -> dict:
    """Profile the input harness over ALL tasks → profile.json (no optimization)."""
    target, ws, proposer_model, opt_cfg, _opt_bench, tasks, mode = _setup(args)
    prof_bench = InstrumentedBenchmark(
        target.make_benchmark(replace(opt_cfg, k=args.profile_k)),
        disk_path=ws / "profile_cache.jsonl")
    backend = make_backend(proposer_model, log_dir=ws / "proposer-logs")
    print(f"=== profile-only: target={args.target} mode={mode} model={args.model} "
          f"k={args.profile_k} N={len(tasks)} ===")
    if args.dry_run:
        print("[dry-run] no spend.")
        return {"target": args.target, "profile_only": True, "n_tasks": len(tasks)}
    seed = resolve_harness(target, backend, ws, cold=args.cold_start, validate=prof_bench.boot_check)
    prof = profile(prof_bench, seed, tasks, k=args.profile_k, save_to=ws / "profile.json")
    h = prof.histogram()
    print(f"pass-rate mean={prof.mean():.3f} | solved={h['solved']} mixed={h['mixed']} "
          f"failing={h['failing']} (N={len(tasks)}) -> {ws / 'profile.json'}")
    return {"target": args.target, "mode": mode, "histogram": h, "mean": prof.mean()}


def run_hillclimb(args) -> dict:
    from .orchestrator import Orchestrator

    target, ws, proposer_model, opt_cfg, opt_bench, tasks, mode = _setup(args)
    test_bench = target.make_benchmark(replace(opt_cfg, k=args.test_k))
    print(f"=== hillclimb: target={args.target} mode={mode} model={args.model} "
          f"localizer={args.localizer} profile={'off' if args.no_profile else f'k={args.profile_k}'} ===")
    if args.dry_run:
        print(f"baseline bar: {target.baseline_score} ({target.baseline_note})")
        print("[dry-run] no spend.")
        return {"target": args.target, "mode": mode, "dry_run": True, "n_tasks": len(tasks)}

    backend = make_backend(proposer_model, log_dir=ws / "proposer-logs")
    seed = resolve_harness(target, backend, ws, cold=args.cold_start, validate=opt_bench.boot_check)
    ok, err = opt_bench.boot_check(seed)
    if not ok:
        raise SystemExit(f"seed harness failed boot_check: {err}")

    prof = None
    if not args.no_profile:
        prof_bench = InstrumentedBenchmark(
            target.make_benchmark(replace(opt_cfg, k=args.profile_k)),
            disk_path=ws / "profile_cache.jsonl")
        prof = profile(prof_bench, seed, tasks, k=args.profile_k, save_to=ws / "profile.json")
        hh = prof.histogram()
        print(f"profile: mean={prof.mean():.3f} | solved={hh['solved']} mixed={hh['mixed']} "
              f"failing={hh['failing']}")
    split = build_split(args, profile=prof, tasks=tasks)
    print(f"split | held_in={len(split.held_in)} regression={len(split.regression)} "
          f"| locked held_out={len(split.held_out)}")

    cfg = _config(args, ws, split)
    orch = Orchestrator(workspace=ws, source_harness=seed, benchmark=opt_bench,
                        backend=backend, config=cfg, part_map=target.part_map(),
                        split=replace(split, held_out=[]))  # optimizer never sees the locked test
    orch.run()

    result = {"target": args.target, "mode": mode, "model": args.model,
              "published_baseline": target.baseline_score,
              **verdict(test_bench, seed, orch.harness, split.held_out,
                        k=args.test_k, sigma2=args.sigma2)}
    (ws / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\n=== VERDICT {args.target} [{mode}] ===")
    print(f"baseline {result['baseline_harness_score']} -> optimized "
          f"{result['optimized_harness_score']} (lift {result['lift']:+.3f} ± "
          f"{result['se']:.3f}, detectable {result['detectable']})")
    print(f"-> {ws / 'result.json'}")
    return result
