"""Bake the nexau runtime into each TB2 task's Docker image (one-time).

AHE's nexau agent expects ``/opt/nexau-venv`` in the container. On E2B that's
pre-built by ``build_templates.py``; on local Docker it's otherwise installed at
runtime *inside every trial* (~40-60s each), which dominates a full 89-task run.

This script appends a guarded nexau-install block to each task's
``environment/Dockerfile`` so ``harbor run --force-build`` bakes ``/opt/nexau-venv``
into the cached image (built once, reused for every trial). Then run both arms
with ``USE_BP_E2B=True`` so the agent's install.sh just *activates* the baked venv.

    # bake the Dockerfiles, then pre-warm the image cache in parallel
    python examples/prebake_nexau.py --tasks fix-git,regex-log,...   # subset
    python examples/prebake_nexau.py --all --build -j 8              # all 89, pre-build

Idempotent: re-running skips Dockerfiles already baked (marker comment).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.benchmark.nexau import DEFAULT_TASK_CACHE  # noqa: E402

MARKER = "# === nexau runtime baked by harness_studio prebake ==="

# Mirrors harbor's install-nexau_saas_e2b.j2, moved to image-build time.
NEXAU_BLOCK = f"""
{MARKER}
USER root
RUN (command -v apt-get >/dev/null 2>&1 && apt-get update && apt-get install -y --no-install-recommends curl git ca-certificates build-essential) || true
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${{PATH}}"
RUN /root/.local/bin/uv venv /opt/nexau-venv --python 3.13 --clear \\
 && /root/.local/bin/uv pip install --python /opt/nexau-venv/bin/python \\
      git+https://github.com/Curry09/NexAU-harbor.git \\
      git+https://github.com/nex-agi/NexAU.git
# === end nexau runtime ===
"""


def task_dirs(cache: Path, names: list[str] | None) -> list[Path]:
    if names:
        out = []
        for n in names:
            m = sorted(cache.glob(f"*/{n}"))
            if not m:
                print(f"[warn] task {n!r} not in cache")
            else:
                out.append(m[0])
        return out
    return sorted(p.parent for p in cache.glob("*/*/task.toml"))


def bake_dockerfile(task_dir: Path) -> str:
    df = task_dir / "environment" / "Dockerfile"
    if not df.is_file():
        return f"SKIP (no Dockerfile): {task_dir.name}"
    text = df.read_text()
    if MARKER in text:
        return f"already baked: {task_dir.name}"
    df.write_text(text.rstrip() + "\n" + NEXAU_BLOCK)
    return f"baked: {task_dir.name}"


def build_image(task_dir: Path) -> str:
    """Pre-warm harbor's image cache: build hb__<task> from the baked Dockerfile."""
    env_dir = task_dir / "environment"
    tag = f"hb__{task_dir.name}"
    r = subprocess.run(["docker", "build", "-t", tag, str(env_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return f"BUILD FAIL {task_dir.name}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}"
    return f"built {tag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=lambda s: [t.strip() for t in s.split(",") if t.strip()], default=None)
    ap.add_argument("--all", action="store_true", help="bake every cached task")
    ap.add_argument("--build", action="store_true", help="also pre-build hb__<task> images to warm the cache")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="parallel docker builds")
    ap.add_argument("--cache", type=Path, default=DEFAULT_TASK_CACHE)
    args = ap.parse_args()

    if not args.all and not args.tasks:
        ap.error("pass --tasks t1,t2 or --all")
    dirs = task_dirs(args.cache, None if args.all else args.tasks)
    print(f"baking {len(dirs)} task Dockerfiles under {args.cache}")
    for d in dirs:
        print("  " + bake_dockerfile(d))

    if args.build:
        print(f"\npre-building {len(dirs)} images ({args.jobs} parallel)...")
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for msg in ex.map(build_image, dirs):
                print("  " + msg)


if __name__ == "__main__":
    main()
