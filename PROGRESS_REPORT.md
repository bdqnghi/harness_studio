# Progress Report — Stochastic Harness Optimizer (SHO / harness_studio)

**Date:** 2026-06-11 · **Branch:** `main` · **Status:** single 3-set protocol hardened after code review; full offline suite green (152 tests); full 2×2 run not yet launched.

> This report supersedes the previous one (the "SHO vs AHE, all‑Gemini" write‑up). That experiment was **retired**: its headline edge was inside k=1 measurement noise. The project pivoted to measuring SHO **against itself** with a noise‑honest protocol. Everything below is the current state.

---

## 0. TL;DR

- **What the system is.** A loop that automatically improves an LLM coding‑agent's *harness* (its setup: instructions, tool descriptions, tool code, middleware, skills, sub‑agents, memory) and then **trustworthily measures** whether the change actually helped — beating the measurement noise instead of being fooled by it.
- **The big lesson that reshaped everything.** Single‑rollout (k=1) pass/fail scoring on these tasks is **noise‑dominated**: it cannot reliably separate two harnesses. So we (a) stopped comparing against AHE on k=1, (b) score the final verdict at **k≥3 rollouts**, and (c) made the accept/reject gate **noise‑aware**.
- **What just shipped (this session).** The evaluation design is one split with three sets — *held‑in pool*, *regression*, *locked test*. A hardening pass now keeps locked task IDs outside the optimizer, calibrates only held-in tasks, grades the audited live harness, validates complete Harbor output, and removes proposer shell execution.
- **It generalizes by size.** The same `choose_split()` produces sane sets whether the benchmark is TB2 (89 tasks) or SWE‑bench (2000): the optimizer's appetite stays a **fixed scoop** (~constant), and the locked test absorbs the surplus, so **more data buys a sharper final number, not a slower optimizer**.
- **It generalizes by model.** Provider and model are explicit inputs. Adapter-specific normalization produces NexAU's bare model plus provider metadata and mini-swe/LiteLLM's `provider/model` form.
- **It generalizes by target harness.** Two harnesses are wired as optimization targets: **nexau** (`code_agent_simple`) and **mini‑swe‑agent** (the whole codebase, injected into the task container).
- **Validation status.** 152/152 tests green. Protocol tests assert that locked tasks never reach calibration or optimization, the live post-audit harness is graded, matrix options propagate, missing Harbor trials fail closed, and all heavy tasks survive a capped test.

---

## 1. The north‑star goal

Build a **noise‑honest, model‑general, harness‑general** self‑improving optimizer, and prove it improved a harness with a number you can trust.

Three properties, each now backed by code:

1. **Noise‑honest** — every "this edit helped" claim survives the actor's stochasticity (k≥3 verdict; power‑sized sets; do‑no‑harm gate).
2. **Model‑general** — the proposer and the actor can be any LLM backbone (LiteLLM).
3. **Harness‑general** — a "harness" is just a directory of files; the optimizer copies, mutates, and re‑runs it. nexau and mini‑swe‑agent are both just such directories.

The headline experiment is SHO **compared to itself**: baseline harness vs. SHO‑optimized harness, per backbone, on a locked held‑out test. (We do **not** re‑run AHE — see §2.)

---

## 2. How we got here (the lessons that shaped the design)

This design is the residue of several hard, specific lessons. They matter because each one is now encoded as a constraint in the code.

### 2.1 k=1 scoring is noise — so stop trusting single rollouts
A probe showed the actor is stochastic enough that a single pass/fail rollout per task cannot separate two harnesses. The old "we beat AHE by +0.015" was **~1/3 of one task — inside the noise band**. Consequences now baked in:
- The **verdict** is scored at `test_k` rollouts (default 3), not 1.
- The **gate** is noise-aware: any result inside the wobble band is unresolved and triggers full-split reruns; task selection never depends on the first noisy result.
- We measure **SHO against itself**, where a clean baseline‑vs‑optimized paired lift is meaningful, instead of against AHE where the cross‑system noise was hopeless.

### 2.2 Percentage‑of‑N splits are wrong at both ends
"Hold out 20%" starves the optimizer at small N and **explodes the per‑round cost** at large N (20% of SWE‑bench = 400 tasks evaluated *every round*). The fix: validation size comes from **statistical power + an affordability cap**, making it ~constant (an SGD mini‑batch). Surplus tasks go to the locked test, which is scored once.

### 2.3 Cross‑validation is the wrong tool here
We briefly adopted 5‑fold CV (detectable ≈ 7.6% on TB2). But in classical ML, CV is cheap because each fold is a cheap *training* run. **Here each fold is a full, multi‑hour SHO optimization** — so 5‑fold CV does the expensive thing **5×**. That cost structure is inverted from where CV pays off. CV appears in **none** of the harness‑optimization papers. We removed it. The replacement: **one** train/test split, and buy statistical power through the cheap lever (test rollouts `k`), not through re‑running the optimizer.

### 2.4 The subset scheme was over‑engineered
At its peak the scheme had **5 piles** (practice, judging, gen, audit, final_exam) and 5 folds — far more complex than the closest prior art (Self‑Harness uses **2** sets). We cut it to **3 sets** (§6.2). This is simpler, cheaper, and still strictly more honest than Self‑Harness (we keep a locked test they don't have).

### 2.5 Heavy tasks must never gate a round
TB2 contains ~14 tasks with 1–3 hour timeouts (e.g. `build-pov-ray`). If even one sits in a set that runs every round, every round waits hours. Rule, now enforced: **heavy tasks (timeout ≥ 1 h) go ONLY into the locked test** (run once), never into any every‑round set. They are still *graded*, just not on the hot path.

---

## 3. Prior art — what we borrow and what we reject

Three 2026 papers attack the same problem. We borrow mechanisms but keep **our noise rigor as the foundation that makes those mechanisms trustworthy** (none of the three does per‑task noise handling our way).

### 3.1 Self‑Harness (arXiv 2606.09498, Shanghai AI Lab)
A fixed model improves its own harness. **Two task sets**: `D_in` (find failures + half the accept test) and `D_ho` (the other half + the reported number). Accept iff `Δ_in ≥ 0 AND Δ_ho ≥ 0 AND max(Δ) > 0`, then merge all accepted edits.
- **Borrowed (shipped):** the **dual‑split acceptance gate** (our `judging` + `regression`), and the **strict‑improvement** rule (a behavioral edit must improve ≥1 split; an additive edit may be neutral‑on‑both).
- **Improved on:** Self‑Harness **reports the set it tuned on** (`D_ho` is part of the gate). We add a **third, fully locked test set** that the optimizer never sees — a cleaner final number.
- **Deferred (phase 2):** verifier‑grounded **failure signatures** φ for clustering + an "addressability" filter; **MergeAccepted**.

### 3.2 AHE / Agentic Harness Engineering (arXiv 2604.25850, Fudan/Peking)
Optimize **and report on the full benchmark — no within‑benchmark split.** Overfitting is checked by **transfer** (frozen harness → a *different* benchmark + other models). pass@1 = **mean over k rollouts** (so it does handle noise, via averaging). Keeps best‑so‑far across iterations.
- **Borrowed:** the k‑rollout averaging idea (our `test_k`); the **transfer** fallback for benchmarks too small to split (our `mode="transfer"`).
- **Why we still split:** transfer needs a second benchmark wired up; for a self‑contained, honest number on one benchmark, a locked test is simpler and direct.

### 3.3 AEvo / Harnessing Agentic Evolution (arXiv 2605.13821, MetaGPT/DeepWisdom)
A meta‑agent edits the **optimizer mechanism** (selection, budget, stopping), not the harness. +26% on TB, ~3× cost.
- **Borrowed lightly (deferred):** an **adaptive run‑plan** — let the segment‑boundary meta‑loop set next‑segment budget/stopping from a windowed accept‑rate (robust knobs only).
- **Rejected:** arbitrary optimizer‑mechanism editing (noise‑amplifying: it "fixes" phantom problems on noisy evidence; heavy; hard to reproduce).

---

## 4. The pipeline, end to end

```
            ┌─────────────────────────────────────────────────────────────┐
  Step 0    │ FREEZE SPLIT from task metadata + conservative σ² prior       │
            │ → locked task IDs are removed before any model/task run        │
            └─────────────────────────────────────────────────────────────┘
                                       │
            ┌─────────────────────────────────────────────────────────────┐
  Step 1    │ CALIBRATE baseline on held-in pool + regression only, at k≥3  │
            │ (or use a provided aggregate noise estimate)                  │
            └─────────────────────────────────────────────────────────────┘
                                       │
            ┌─────────────────────────────────────────────────────────────┐
  Step 2    │ OPTIMIZE (per round, for `rounds` rounds):                    │
            │   sample round_size from pool → run actor → find a failure    │
            │   → diagnoser blames 1 of 7 components → strategist proposes   │
            │   competing edits → shell enforces invariants                 │
            │   → DUAL-SPLIT GATE: keep edit iff it doesn't regress judging  │
            │     (a stable slice of the pool) AND doesn't regress           │
            │     regression, and clears the residual noise threshold        │
            └─────────────────────────────────────────────────────────────┘
                                       │
            ┌─────────────────────────────────────────────────────────────┐
  Step 3    │ VERDICT: score baseline vs audited LIVE harness on locked test │
            │ at test_k; the optimizer never receives the locked task IDs    │
            │ → paired per-task lift ± standard error, vs the detectable     │
            │   floor. This is the one honest number.                       │
            └─────────────────────────────────────────────────────────────┘
```

---

## 5. The optimization space — exactly 7 components

A harness is an open codebase. Its editable surface is labeled into **seven component types** (`studio/parts.py`, `PartType`). This is the entire optimization space — it was **not** expanded:

| # | PartType | What it is |
|---|---|---|
| 1 | `INSTRUCTIONS` | system/instance prompts, guidance prose |
| 2 | `TOOL_DESCRIPTIONS` | the text the model reads about each tool |
| 3 | `TOOL_CODE` | the implementation of the tools |
| 4 | `MIDDLEWARE` | ret/ output‑capping / safe‑exec / request shaping |
| 5 | `SKILLS` | reusable skill files |
| 6 | `SUBAGENTS` | sub‑agent configuration |
| 7 | `MEMORY` | message‑history / memory handling |

A **PartMap** labels *which files of a given harness* implement each of these 7 types; everything unmapped is `do_not_touch` (packaging, entrypoints, version). Two harnesses → two PartMaps → the **same 7 buckets**, different files. nexau's `code_agent.yaml`/tools map into the 7; mini‑swe's `config/*.yaml`, `models/`, `environments/`, `agents/default.py` map into the same 7.

**vs Self‑Harness:** Self‑Harness imposes **no fixed component list** — its proposer edits arbitrary regions of the harness code; its only structure is on *failures* (signature clustering). We impose a schema on the *edit surface*, which buys: (a) **targeted blame** ("this is a `TOOL_CODE` failure"), (b) **per‑component edit budgets** + a family‑map meta‑loop, (c) a **protected boundary** (`do_not_touch` plumbing). We are **more structured, not bigger**.

---

## 6. The evaluation split — the part rebuilt this session

### 6.1 The principle: a fixed appetite, like an SGD mini‑batch
The optimizer's per‑round demand for tasks does **not** grow with the benchmark. Like SGD's batch size, it's a fixed scoop just large enough for a usable signal. Two of the three numbers are fixed; only the locked test grows.

### 6.2 The three sets

| Set | Code field | Job | Sizing |
|---|---|---|---|
| **Held‑in pool** | `practice` | Variety pool; each round samples `round_size` from it to find failures and first‑check an edit | `clamp(pool_mult·round_size, round_size, pool_cap)` → ~**128** |
| **Regression** | `regression` | Disjoint do‑no‑harm 2nd set; an edit must not hurt it (= Self‑Harness `D_ho`) | `clamp(power_n(δ_round), reg_floor, reg_cap)` → ~**16–32** |
| **Locked test** | `final_exam` | Never passed to calibration or `Orchestrator`; used only by the final paired verdict at `test_k` | **everything else**, incl. ALL heavy tasks; a cap must still retain every heavy task and `test_floor` |

Plus two derived slices of the pool: `judging` (a **stable** power‑sized gate slice ⊆ pool, scored old‑vs‑new every round) and `audit` (a small slice the deep auditor re‑checks). Disjointness enforced: `test` ⟂ everything; `regression` ⟂ `pool`; `judging`,`audit` ⊆ `pool` (overlap intended — this is `D_in` doing double duty).

### 6.3 The algorithm (`studio/components/splitter.py::choose_split`)
```
HEAVY  = timeout ≥ heavy_sec   → test-only
LIGHT  = the rest
reg_n     = clamp(power_n(σ², δ_round, opt_k), reg_floor=16, reg_cap=32)
pool_n    = clamp(pool_mult·round_size, round_size, pool_cap=256)     # ~128
n_judging = clamp(power_n(σ², δ_round, opt_k), val_floor=8, round_size)
# shrink-to-fit for small N: pool shrinks first (→ round_size), then regression (→ reg_floor)
# test must reach test_floor (heavy tasks count toward it)
# if still can't seat an honest test → mode="transfer" (optimize on all, verify elsewhere)
test = HEAVY + (LIGHT not in pool ∪ regression)
# if capped: keep ALL HEAVY + stratified light sample; reject an impossible cap
```

### 6.4 What it produces across benchmark sizes (validated)

| Benchmark | N | held‑in pool | regression | locked test | each round runs | test detectable @k=3 |
|---|---|---|---|---|---|---|
| **TB2** | 89 (14 heavy) | 32 | 32 | **25** (14 heavy + 11 light) | 32 | ~0.143 |
| SWE‑bench Lite | 300 | 128 | 32 | **140** | 32 | ~0.060 |
| SWE‑bench | 2000 | 128 | 32 | **500** (cap; pool ~1840 locked) | 32 | ~0.032 |
| tiny | 20 | — | — | — | — | **transfer mode** |

The held‑in scoop barely moves (32→128) while N grows 22×; the locked test absorbs the surplus and the detectable effect sharpens (0.143 → 0.032). Exactly the intended behavior.

### 6.5 Calibration & power math (`studio/components/calibration.py`)
- The split is frozen before measured calibration. The baseline then runs on held-in pool + regression only at `calibration_k≥3`; locked outcomes are never used for split selection or optimization.
- `σ² = mean_t p_t(1−p_t)` uses a finite-sample correction for rates estimated from repeated rollouts and is clamped to [0.01, 0.25].
- `power_n(σ², z, δ, k) = ⌈z²·2σ² / (k·δ²)⌉` — tasks needed to resolve effect δ. `detectable_delta(n, …)` is its inverse.
- Free `task.toml` metadata (no eval): per‑task `timeout_sec` (heavy detection) and `[metadata].difficulty` (cold‑start stratification before any run).
- **Provided-baseline path:** the split is still frozen from metadata + `--sigma2-prior`. Then `--baseline-json {task: rate}` may skip calibration only when every held-in task is present; extra locked-task entries are ignored. Aggregate `--baseline-score X` and binary-only rates require `--baseline-sigma2`.

### 6.6 The dual‑split gate (`studio/components/gate.py`)
For each candidate edit, score gain on `judging` and on `regression`:
- Clear regression on either split (`< −wobble`) → **reject**.
- **Behavioral** edits must beat wobble (or the reduced residual threshold after reruns). A neutral edit is rejected.
- Only diffs that exclusively add files are **additive** and may be neutral-on-both; modifying or deleting any existing file uses the behavioral rule.
- Any split result in `[−wobble, +wobble]` is borderline. The complete predeclared split is rerun, avoiding first-rollout selection bias.
- The single‑split legacy path (no `regression` set) is preserved for backward compat.

### 6.7 Model generality (`studio/backends/llm.py`, `factory.py`)
- `LLMBackend` wraps `litellm.completion(...)` for the proposer's Tier‑A tool loop and Tier‑B JSON, with the same retry + thinking‑guard + cost accounting as the original Gemini backend.
- `make_backend(model, …)` routes: `claude-cli/...` → `ClaudeCLIBackend`; anything else → `LLMBackend`.
- `--provider` + `--model` configure a custom cell. NexAU receives explicit provider metadata; mini-swe and LiteLLM receive normalized `provider/model`.
- Tier-A API backends expose workspace-jailed file tools but no arbitrary shell command. The trusted structural check performs validation after editing.

### 6.8 Second harness target (`studio/benchmark/mini_swe.py` + harbor patch)
- `MiniSweBenchmark` runs `harbor run --agent mini-swe-agent --env docker -m <litellm-model>`; `mini_swe_part_map()` maps mini‑swe's files into the 7 components.
- **Injection (proven):** the optimizer's *mutated* mini‑swe copy is uploaded to the container (`MSWEA_HARNESS_DIR` → `/mswea-harness`), and the install script does `uv tool install /mswea-harness` so `mini` runs **our** code+config. Smoke test confirmed a sentinel injected into our system prompt reaches the container trajectory (4 artifacts), and mini‑swe solved `overfull-hbox`.
- A preflight now verifies both Harbor patch markers before every real mini-swe run and fails closed if injection is absent.

### 6.9 The experiment: a 2×2 self‑harness matrix
`{nexau, mini‑swe} × {gemini‑3.5‑flash, gpt‑5.4}`, each cell a **pure self‑harness** run (the backbone is *both* proposer and actor, improving its own harness). Gemini‑key cells and OpenAI‑key cells run on **separate quotas → genuinely parallel, no 429 contention**; the two backbones are **never combined** — each is an independent finding (which backbone self‑improves more). Driver: `examples/tb2_self_compare.py` (`--matrix`).

---

## 7. Implementation status (file by file)

**Rewritten / added this session:**
- `studio/components/splitter.py` — **rewrote**: removed CV machinery (`choose_eval_plan`, `dynamic_split`, `_make_kfold`, `_carve_optimization`); added **`choose_split`** (single 3‑set split, transfer fallback); kept `power_n`, `detectable_delta`, `_strata`, `_stratified_sample`; `TaskSplit.gen → regression`; new `SplitPlan` fields (`n_pool`, `n_judging`, `n_regression`, `n_test`, `detectable_round`, `detectable_final`, `recommend`).
- `studio/components/gate.py` — dual-split gate plus full-split borderline reruns and residual-noise threshold.
- `studio/orchestrator.py` — gate uses regression tasks, measures wobble on both gate splits, and classifies additions from the actual diff.
- `studio/config.py` — `EvalPlanConfig` rewritten for the new knobs (`round_size`, `reg_floor`, `reg_cap`, `pool_mult`, `pool_cap`, `test_floor`, `test_budget_cap`, `opt_k`, `test_k`, `delta_round`).
- `examples/tb2_self_compare.py` — freezes the split before calibration, strips locked IDs before constructing `Orchestrator`, grades `orch.harness`, normalizes configurable provider/model input, and forwards all matrix options.
- `studio/benchmark/{kira,nexau,mini_swe}.py` — complete-trial validation; nonzero Harbor exits and missing trials now raise instead of becoming false task failures.
- `studio/backends/{gemini,claude_cli}.py` — arbitrary proposer shell execution removed.
- `studio/components/calibration.py` — docstring pointer updated to `choose_split`.
- `tests/test_power_split.py` — **rewrote** for `choose_split` (constant held‑in across N; test grows; TB2 single holdout with heavy‑in‑test; regression disjoint; stratified; sigma‑floor; transfer for tiny N; determinism).
- `tests/test_dual_split_gate.py` — renamed `gen` → `regression`.
- `tests/test_dynamic_split.py` — **removed** (tested the deleted `dynamic_split`).

**Built in prior sessions (still current):** `studio/backends/llm.py` + `studio/backends/factory.py` (LiteLLM generality); `studio/benchmark/mini_swe.py` + `mini_swe_part_map()`; provider→api_type map in `studio/benchmark/nexau.py`; `studio/components/calibration.py`; harbor patches (`install-mini-swe-agent.sh.j2`, `mini_swe_agent.py` `setup()`); curated `artifacts/mini_swe_harness/`; tests `test_calibration.py`, `test_make_backend.py`, `test_mini_swe_partmap.py`.

**Test status:** `152/152 green` via `.venv/bin/python -m pytest -q`.

---

## 8. What is NOT done / deferred (phase 2)

- **The 2×2 full run** — wired and dry‑run‑validated, **not yet launched** (needs Docker + compute go‑ahead + the two open decisions in §9).
- **Failure signatures φ + addressability filter** (Self‑Harness C3/C4) — the retrieval key for a future edit library.
- **Edit library** keyed by φ (seed the proposer with proven general edits; no extra rollouts).
- **Adaptive run‑plan** (AEvo C6) — meta‑loop sets next‑segment budget/stopping from windowed accept‑rate.
- **Transfer‑mode wiring** — `choose_split` returns `mode="transfer"` for tiny N; the driver now refuses to optimize or emit a verdict until a transfer benchmark is configured.
- **Per‑segment validation resampling**, paired‑SPRT gating, IRT/tinyBenchmarks compression for huge N, adaptive `k`.
- **MergeAccepted** (merge all gate‑passing edits per round).

---

## 9. Open decisions (need user input before launch)

1. **`test_k` for the verdict** — k=5 (≈10% detectable on TB2's 25‑task test, ~8 h/cell) vs k=9 (≈7.5%, ~12 h/cell) vs k=3 (≈14%, cheapest). Trade compute for the smallest lift we can trust.
2. **Direction** — **B (shipped): minimal 3‑set split** with a locked test, vs **A: AHE‑style** (optimize on all 89, prove generalization by transfer to a second benchmark). B is implemented; A would need a transfer benchmark wired.

---

## 10. How to run

```bash
cd /home/nghibui/codes/harness_studio

# unit tests (use the venv python; bare `python` is absent)
.venv/bin/python -m pytest -q

# preview the plan for one cell — no Docker, no spend
.venv/bin/python examples/tb2_self_compare.py --harness nexau --backbone gemini --dry-run

# real single cell with a PROVIDED aggregate baseline (skips calibration)
.venv/bin/python examples/tb2_self_compare.py --harness mini-swe --backbone gpt-5.4 \
    --baseline-score 0.62 --baseline-sigma2 0.20

# the full 2x2 (gemini-key + openai-key cells in parallel, one harness per wave)
.venv/bin/python examples/tb2_self_compare.py --matrix --test-k 5
```
Env for real runs: `AHE_DIR`, `HARBOR_TASK_CACHE`, `USE_BP_E2B=True`, provider keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`).

---

## 11. Immediate next steps

1. Confirm the two §9 decisions (`test_k`; stay on direction B).
2. Launch the 2×2 via `--matrix` (or one cell first as a cheap smoke), monitor to the lift table.
3. Read the per‑cell `lift.json` → a 2×2 table of (lift ± SE) vs the detectable floor; the verdict is *which backbone × which harness self‑improves most, beyond noise*.
4. Then phase 2: failure signatures → edit library.

---

## 12. Key file index

| Concern | File |
|---|---|
| 7 component types + PartMap | `studio/parts.py` |
| **The 3‑set split** | `studio/components/splitter.py` (`choose_split`) |
| Calibration + power math | `studio/components/calibration.py` |
| **Dual‑split gate** | `studio/components/gate.py` |
| Config knobs | `studio/config.py` (`EvalPlanConfig`) |
| Orchestrator (the loop) | `studio/orchestrator.py` |
| Model generality | `studio/backends/llm.py`, `studio/backends/factory.py` |
| nexau target + api_type map | `studio/benchmark/nexau.py` |
| mini‑swe target + injection | `studio/benchmark/mini_swe.py`, harbor patches |
| **The experiment driver** | `examples/tb2_self_compare.py` |
| Memory (cross‑session facts) | `~/.claude/.../memory/` (`tb2-restart-plan`, `harness-opt-papers`, `tb2-measurement-noise`, …) |
