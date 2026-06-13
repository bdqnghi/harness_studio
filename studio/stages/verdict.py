"""Step 5 — verdict: grade the seed vs the optimized harness on the locked
held_out set.

This is the only place the held_out (locked) test is scored. It returns a paired
per-task lift ± standard error plus a detectable-delta estimate, so the final
number is honest about whether the gain exceeds the benchmark's noise.
"""

from __future__ import annotations

import statistics

from .split import detectable_delta


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
