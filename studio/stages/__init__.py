"""The SHO steps, as modules.

The spine (``studio/pipeline.py``) wires five steps in order:

1. **resolve** the round-0 harness — ``Target.resolve_seed`` (``studio/targets.py``);
   cold start uses ``optimize/strategist.build_harness``.
2. **profile** the seed over all tasks — :mod:`studio.stages.profile`.
3. **split** into held_in / regression / held_out — :mod:`studio.stages.split`.
4. **optimize** (the per-round hypothesis-tree loop) — :mod:`studio.stages.optimize`.
5. **verdict** — grade seed vs optimized on the locked held_out — :mod:`studio.stages.verdict`.
"""
