"""Score a harness directory on a fixed TB2 task list (the head-to-head metric).

Reusable for: the baseline (bare code_agent_simple), the AHE-evolved workspace,
and harness_studio's evolved best. All three are scored on the SAME locked
held-out pile (``tb2_config.final_tasks()``) so the numbers are comparable.

    python examples/tb2_score.py <harness_dir> --label baseline
    python examples/tb2_score.py <ahe_workspace> --label ahe --out /tmp/ahe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.tb2_config import final_tasks  # noqa: E402
from studio.benchmark.nexau import NexauBenchmark  # noqa: E402
from studio.harness import Harness  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("harness_dir", type=Path)
    ap.add_argument("--label", default="harness")
    ap.add_argument("--tasks", type=lambda s: [t.strip() for t in s.split(",") if t.strip()], default=None)
    ap.add_argument("--model", default="gemini-3.5-flash", help="actor model (must match the run for fairness)")
    ap.add_argument("--k", type=int, default=1, help="rollouts/task (k>1 lowers binary variance)")
    ap.add_argument("--n-concurrent", type=int, default=4)
    ap.add_argument("--timeout-multiplier", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tasks = args.tasks or final_tasks()
    harness = Harness(args.harness_dir)
    bench = NexauBenchmark(
        real=True, tasks=tasks, k=args.k, model=args.model,
        n_concurrent=args.n_concurrent, timeout_multiplier=args.timeout_multiplier,
    )
    print(f"[{args.label}] scoring {harness.root} on {len(tasks)} tasks: {tasks}", flush=True)
    t0 = time.time()
    scores = bench.run(harness, tasks, run_idx=0)
    elapsed = (time.time() - t0) / 60
    pass_rate = sum(scores.values()) / len(scores) if scores else 0.0
    result = {
        "label": args.label,
        "harness_dir": str(harness.root),
        "pass_rate": round(pass_rate, 4),
        "n_tasks": len(tasks),
        "k": args.k,
        "per_task": scores,
        "elapsed_min": round(elapsed, 1),
    }
    print(f"\n[{args.label}] PASS RATE = {pass_rate:.3f} ({sum(1 for v in scores.values() if v>=1.0)}/{len(scores)}) "
          f"in {elapsed:.1f} min", flush=True)
    for t, v in scores.items():
        print(f"    {'PASS' if v >= 1.0 else 'fail'}  {t}  ({v})", flush=True)
    out = args.out or Path(f"/tmp/tb2_score_{args.label}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[{args.label}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
