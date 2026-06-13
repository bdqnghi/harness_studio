#!/usr/bin/env python
"""Target-agnostic hill-climb driver — a thin CLI over ``studio.pipeline``.

Resolves a registered ``Target``, then runs the pipeline:
resolve harness (warm shipped / cold-generated) → profile the input harness over
the benchmark → difficulty-stratified split → optimize (tree + net acceptance +
localizer) → grade on the locked held_out.

  # preview (no spend)
  python examples/hillclimb.py --target tau2-airline --dry-run

  # profile only: per-task pass/fail + trajectories -> profile.json
  python examples/hillclimb.py --target tau2-airline --profile-only --profile-k 2 \
      --model gpt-4.1-mini --user-model gpt-4.1-mini

  # cold-start hill-climb (generate a harness, profile it, then climb)
  python examples/hillclimb.py --target tau2-retail --cold-start --model gpt-4.1-mini
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import studio.targets_builtin  # noqa: E402,F401  (populates the Target registry)
from studio import pipeline  # noqa: E402
from studio.targets import list_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help=f"one of: {list_targets()}")
    ap.add_argument("--model", default="gpt-4.1", help="agent model (litellm)")
    ap.add_argument("--proposer-model", default=None, help="override proposer model (default: --model)")
    ap.add_argument("--user-model", default=None, help="tau2 user-simulator model")
    ap.add_argument("--cold-start", action="store_true", help="generate a harness even if a seed ships")
    # profiling
    ap.add_argument("--profile-only", action="store_true",
                    help="run the input harness over ALL tasks once -> profile.json "
                         "(per-task pass/fail + trajectories); no optimization")
    ap.add_argument("--profile-k", type=int, default=2, help="rollouts/task for profiling")
    ap.add_argument("--no-profile", action="store_true",
                    help="skip profiling; use a blind random split instead of stratified")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="cap tasks to a deterministic seeded sample (0=all); for "
                         "huge domains like tau2-telecom (2285 tasks)")
    # optimizer
    ap.add_argument("--localizer", choices=("off", "inline", "agentic", "auto"),
                    default="auto", help="evidence-grounded context localization (off=diagnosis-only)")
    ap.add_argument("--strict-acceptance", action="store_true",
                    help="require EACH slice (held_in AND regression) to not-regress; "
                         "default is net pooled gain")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workspace", default="/tmp/sho_hillclimb")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--segment-length", type=int, default=2)
    ap.add_argument("--hypotheses", type=int, default=4)
    ap.add_argument("--round-size", type=int, default=16)
    ap.add_argument("--noise-floor-runs", type=int, default=3)
    ap.add_argument("--borderline-runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--opt-k", type=int, default=1)
    ap.add_argument("--test-k", type=int, default=3)
    ap.add_argument("--n-concurrent", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma2", type=float, default=0.2, help="noise prior for the detectable-delta report")
    ap.add_argument("--held-in", type=int, default=16,
                    help="held-in scoop the acceptance scores on each round (keep small/cheap)")
    ap.add_argument("--reg", type=int, default=10, help="regression (do-no-harm) set size")
    ap.add_argument("--held-out", type=int, default=24,
                    help="locked test cap (0=all surplus); each is graded at test_k twice")
    args = ap.parse_args()
    if args.profile_only:
        pipeline.profile_only(args)
    else:
        pipeline.run_hillclimb(args)


if __name__ == "__main__":
    main()
