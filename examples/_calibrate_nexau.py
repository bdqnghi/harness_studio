"""One-task calibration: prove NexauBenchmark scores a real TB2 task via harbor.

Runs the bare code_agent_simple harness on `fix-git` (which AHE's smoke solved at
100% in ~16 min) through our adapter, end-to-end against real harbor + Docker +
gpt-5.4. A reward of 1.0 validates BOTH the harbor invocation and the reward
parsing. Also measures real wall-clock per eval (the overnight budget driver).

    python examples/_calibrate_nexau.py [task_name]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio.benchmark.nexau import DEFAULT_AHE_DIR, NexauBenchmark  # noqa: E402
from studio.harness import Harness  # noqa: E402

task = sys.argv[1] if len(sys.argv) > 1 else "fix-git"
harness = Harness(DEFAULT_AHE_DIR / "agents" / "code_agent_simple")
bench = NexauBenchmark(real=True, tasks=[task], k=1, n_concurrent=1, timeout_multiplier=3.0)

print(f"[calibrate] task={task}  harness={harness.root}", flush=True)
print(f"[calibrate] boot_check={bench.boot_check(harness)}", flush=True)
t0 = time.time()
scores = bench.run(harness, [task], run_idx=0)
elapsed = time.time() - t0
out = {"task": task, "score": scores.get(task), "elapsed_min": round(elapsed / 60, 1)}
print(f"[calibrate] DONE {json.dumps(out)}", flush=True)
Path("/tmp/nexau_calibration.json").write_text(json.dumps(out, indent=2))
