"""Shared definition of the harness_studio-vs-AHE Terminal-Bench 2 head-to-head.

Single source of truth so both arms are provably comparable:

* the SAME input harness  -> AHE's ``agents/code_agent_simple/`` (a bare nexau agent)
* the SAME actor model     -> gpt-5.4 (set via the LLM_* env, an env reference in
  the yaml, so no harness edit can change the model)
* the SAME locked held-out -> ``final_tasks()`` (scored for baseline, AHE-best,
  and ours-best; never used for optimization by either arm)
* the SAME optimization pool-> ``opt_tasks()`` (audit+judging+practice); AHE
  optimizes on exactly these, harness_studio optimizes on the full 16 but holds
  out the same final pile via the seeded splitter.

The 16 tasks are a fixed, difficulty-balanced subset of the 99 locally-cached
TB2 tasks (all Docker-compatible). Easy tasks give signal; hard tasks give the
bare single-shell-tool agent real headroom to improve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.components.splitter import TaskSplit, split_tasks  # noqa: E402
from studio.config import PileConfig  # noqa: E402

SEED = int(os.environ.get("TB2_SEED", "7"))

# 16 fixed tasks (4 easy / 9 medium / 3 hard), all in ~/.cache/harbor/tasks.
_FULL_TASKS: list[str] = [
    # easy
    "fix-git",
    "cobol-modernization",
    "overfull-hbox",
    "prove-plus-comm",
    # medium
    "extract-elf",
    "sqlite-db-truncate",
    "query-optimize",
    "regex-log",
    "git-leak-recovery",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "large-scale-text-editing",
    "headless-terminal",
    # hard
    "fix-code-vulnerability",
    "cancel-async-tasks",
    "write-compressor",
]

# A smaller shared set can be selected via env (for a feasibility-limited local
# head-to-head) WITHOUT changing the committed full set — both arms read this,
# so the held-out pile and optimization pool stay consistent across arms.
#   export TB2_TASKS="fix-git,cobol-modernization,..."  TB2_FINAL=3 TB2_AUDIT=1 TB2_JUDGING=2
if os.environ.get("TB2_TASKS"):
    TASKS = [t.strip() for t in os.environ["TB2_TASKS"].split(",") if t.strip()]
    PILES = PileConfig(
        practice=max(1, len(TASKS) - int(os.environ.get("TB2_FINAL", "3"))
                     - int(os.environ.get("TB2_AUDIT", "1")) - int(os.environ.get("TB2_JUDGING", "2"))),
        judging=int(os.environ.get("TB2_JUDGING", "2")),
        audit=int(os.environ.get("TB2_AUDIT", "1")),
        final_exam=int(os.environ.get("TB2_FINAL", "3")),
    )
else:
    TASKS = _FULL_TASKS
    # final_exam (6) is the locked held-out pile; audit(3)+judging(3)+practice(4)=10
    # form the optimization pool both arms may learn from.
    PILES = PileConfig(practice=4, judging=3, audit=3, final_exam=6)


def split() -> TaskSplit:
    return split_tasks(TASKS, PILES, seed=SEED)


def final_tasks() -> list[str]:
    """The locked held-out pile — the head-to-head metric is pass-rate on these."""
    return split().final_exam


def opt_tasks() -> list[str]:
    """The optimization pool AHE's arm runs on (audit+judging+practice)."""
    s = split()
    return s.audit + s.judging + s.practice


if __name__ == "__main__":
    s = split()
    print(f"seed={SEED}  total={len(TASKS)}")
    print(f"final_exam (LOCKED held-out, {len(s.final_exam)}): {s.final_exam}")
    print(f"optimization pool ({len(opt_tasks())}): {opt_tasks()}")
    print(f"  audit   ({len(s.audit)}): {s.audit}")
    print(f"  judging ({len(s.judging)}): {s.judging}")
    print(f"  practice({len(s.practice)}): {s.practice}")
