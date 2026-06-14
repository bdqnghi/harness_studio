# SHO workflow — how a run goes from input to verdict

The spine is `studio/pipeline.py`, which wires five steps:
**resolve → profile → split → optimize → verdict.** This doc walks each one and
names the code that implements it.

> Terminology note: what older notes called the *gate* is now the **acceptance
> check** (`stages/optimize/evaluate/acceptance.py`), and *wobble* is now the
> **noise floor** (`stages/optimize/evaluate/noise_floor.py`).

---

## Inputs

- **`--target`** — a registered `Target` (`tau2-retail`, `qa-hotpot`, …). It knows:
  how to build the benchmark, the seed harness (or `None` for cold-start), the
  part-map (which files are editable), and the published baseline.
- **`--model`** — the agent that *runs the tasks* (the model whose harness we're tuning).
- **`--proposer-model`** — the coding agent that *edits the harness* (defaults to `--model`).
- **Knobs** — `--rounds`, `--held-in`/`--reg`/`--held-out`, `--profile-k`/`--opt-k`/`--test-k`,
  `--localizer`, `--strict-acceptance`, `--max-tasks`, `--seed`.

The **harness** = a directory of editable text files (tau2: `policy.md`; QA:
`system_prompt.md`). That directory **is** the thing being optimized.

---

## Step 0 — Setup (`pipeline._setup`)

Build the benchmark from the target + config, list all tasks, apply the
`--max-tasks` deterministic cap, and decide warm vs cold start.

## Step 1 — Resolve the round-0 harness (`pipeline.resolve_harness` → `Target.resolve_seed`)

- **Warm start**: copy the benchmark's shipped seed harness.
- **Cold start** (no seed, or `--cold-start`): run the coding agent on an empty
  workspace to *generate* a harness from the `ColdStartBrief`, retrying until
  `boot_check` passes.

## Step 2 — Profile the seed (`stages/profile.py::profile_harness`)

Run the seed over **all** tasks once (chunked, at `--profile-k` rollouts each) →
`profile.json = {task_id: pass_rate}`, where `pass_rate` is the mean score over k
rollouts. Side effect: failure evidence for the failing tasks lands in
`benchmark.evidence_store`.

> Caveat: `pass_rate` means "fraction of rollouts passed" for **binary** graders
> (tau2, gsm8k), but "mean partial-credit" for **continuous** ones
> (hotpot/searchqa F1) — the bins below treat them the same.

## Step 3 — Difficulty-stratified split (`stages/split.py::stratified_split`)

Bin each task: **solved** (≥0.8), **failing** (≤0.2), **mixed** (between). Then
carve three disjoint sets, in this order:

1. **held_out** (locked test) — a representative proportional sample across all
   three bins, taken first so it's unbiased. Size `--held-out` (24).
2. **regression** (do-no-harm guard) — reliably-solved tasks not in held_out.
   Size `--reg` (10).
3. **held_in** (the acceptance check's working set) — failing → mixed → solved
   priority, so the optimizer gets learnable failures first. Size `--held-in` (16).

(`--no-profile` → `random_split`: blind seeded shuffle, no difficulty awareness.)

## Step 4 — Optimize (`stages/optimize/orchestrator.py::Orchestrator.run`, tree-only; never sees held_out)

**Once at start:** measure the **noise floor** (`evaluate/noise_floor.py`) by
re-running held_in + regression a few times. This is the bar a real gain must clear.

**Then each round (`_round_tree`):**

1. **Run** held_in → collect failing tasks (`diagnose/runner.py`).
2. **Diagnose** (`diagnose/diagnoser.py`) — cluster failures into patterns with a
   blamed part; drop non-addressable ones.
3. **Route + select** — place patterns onto the hypothesis tree (directions →
   hypotheses) and pick one hypothesis to try (`propose/` — Thompson-style;
   falsified ideas never retried, noise-killed ones retried boundedly).
4. **Localize** (`edit/localizer.py`; `--localizer auto/inline/agentic/off`) — from
   the failure evidence, identify the causal files/regions to edit.
5. **Implement** (`edit/strategist.py::implement_hypothesis`) — the coding agent
   edits the localized files → a candidate harness; structural/shell check
   (+ optional repair).
6. **Acceptance check** (`evaluate/acceptance.py::AcceptanceCheck.evaluate`) — score
   candidate vs current on held_in **and** regression, pooled. Accept iff net pooled
   gain exceeds the noise floor (regression is non-veto by default; `--strict-acceptance`
   makes it a hard veto). Borderline → extra re-runs to beat noise.
   - **Accept** → candidate becomes the live harness.
   - **Reject** → classify (falsified / rejected-as-noise) and distill an
     **insight** (`propose/insight.py`) back into the tree node so the next ideas
     are smarter.
7. Periodically **deep-audit** (`evaluate/deep_auditor.py`, re-check held_in) and
   update tree statistics.

## Step 5 — Verdict (`stages/verdict.py::verdict`)

Grade seed vs optimized on the **locked held_out** at `--test-k`: paired per-task
lift ± standard error, plus a detectable-delta estimate → `result.json`.

---

## What persists

`profile.json`, `result.json`, `idea_tree.json`/`tree.md`, `candidates/`, `best/`,
`evidence.jsonl`, `score_cache.jsonl`, `progress.jsonl`. Raw trajectories are
discarded — only scores + compact failure evidence survive.

---

**One-line mental model:** resolve a harness → profile it to map task difficulty →
split into a failure-heavy training set, a do-no-harm guard, and a locked test →
loop {find failures → diagnose → pick a hypothesis off the tree → localize → edit →
noise-aware acceptance check} → grade the before/after on the locked test.
