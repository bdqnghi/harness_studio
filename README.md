# harness_studio

A self-evolving **harness optimizer**. It automatically improves an AI agent's
*harness* — the scaffolding around a frozen model: instructions, tool
descriptions, tool code, middleware, skills, sub-agent config, and memory — by
repeatedly proposing coordinated edits, testing them against a real benchmark
with a **noise-aware acceptance check**, and keeping only what genuinely helps. A slower
**meta-loop** then revises *how* edits are proposed, so the search keeps escaping
plateaus.

It combines three ideas (see `PRD_harness_optimizer.md`): SkillOpt's optimizer
discipline, AHE's typed multi-component harness, and AEVO's two-speed
meta-evolution. The keep/reject decision is always objective arithmetic on real
scores — never an AI judge.

## Design in one breath

Three pluggable **seams** keep the loop testable for free:

| Seam | Real implementation | Test implementation |
|---|---|---|
| `Backend` — how AI helpers run | `ClaudeCLIBackend` (subprocess `claude -p`) | `MockBackend` (scripted, deterministic) |
| `Benchmark` — how a harness is scored | `KiraBenchmark` (Terminus-KIRA) | `ToyBenchmark` (known optimum + injected noise floor) |
| `Harness` — the thing optimized | a real codebase of files | the toy harness |

The **trust boundary** is a code guarantee: AI helpers only ever *propose*; only
the `AcceptanceCheck` writes scores and mutates the harness, and the acceptance check never receives a
`Backend`. No AI component can reach the evaluator.

## Layout

```
studio/
  pipeline.py         the spine: resolve → profile → split → optimize → verdict
  config.py schemas.py
  targets.py targets_builtin.py   step 1 "resolve" = Target.resolve_seed (+ registry)
  core/       harness.py parts.py state.py observe.py evidence.py   (shared vocabulary)
  backends/   base.py mock.py llm.py llm_loop.py factory.py
  benchmark/  base.py toy.py tau2.py qa.py qa_suites.py instrument.py ...
  stages/     the 5 steps
    profile.py  split.py  verdict.py
    optimize/   orchestrator.py + the per-round loop:
                runner diagnoser idea_tree ideator insight localizer
                strategist acceptance noise_floor snapshotter mapper ...
  skills/     strategist/SKILL.md
examples/   hillclimb.py   (target-agnostic driver; --profile-only / --dry-run)
tests/      unit per module + test_tree_loop.py (integration)
```

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

# unit + integration tests (free, deterministic — no API calls)
pytest

# end-to-end on the toy target with the deterministic mock proposer
python examples/run_toy.py --backend mock

# end-to-end on the toy with a real `claude -p` coding agent as the proposer
python examples/run_toy.py --backend claude --rounds 3
```

## Status — all milestones complete ✅

- **M0** — skeleton + seams + toy + noise-aware acceptance check (PRD Phase 0)
- **M1** — typing: Mapper, 7 parts, per-part budgets, structural pre-acceptance check (validated on real Terminus-KIRA)
- **M2** — strategy unit: Diagnoser, competing Strategists, Reviewer, Ranker (top-1 fall-through)
- **M3** — two-speed meta-loop: family map, Meta-agent, deep audit, segments
- **M4** — real KIRA/harbor adapter, health monitor, reward-hack defense, caching, cost instrumentation

Run `pytest` (all deterministic, no API calls). The real `claude -p` paths are
exercised by `examples/run_toy.py --backend claude` and `examples/run_kira_smoke.py`.

See `PRD_harness_optimizer.md` for the full spec and
`/home/nghibui/.claude/plans/oh-i-forgot-you-playful-taco.md` for the plan.
