# Progress Report — harness_studio vs AHE on Terminal-Bench 2

**Branch:** `main` (all work merged; the `tb2-nexau-headtohead` feature branch was deleted). · **Status:** method built/validated/strengthened and committed; one side of the head-to-head is **validly measured** (ours/baseline = 0.667 on the held-out 3), the AHE held-out number was **not captured** before the run was stopped — so **no final ours-vs-AHE verdict yet**.

---

## TL;DR (honest)

- We **wired harness_studio to optimize AHE's exact input harness** (`code_agent_simple`, a bare nexau agent) on Terminal-Bench 2, scored by the **identical** `harbor run --agent nexau` path with the **same frozen actor** (gpt-5.4). Calibrated end-to-end: `fix-git` = 1.0 in 2.4 min.
- We **gave our optimizer AHE-equivalent freedom** (it can now *add* tools/middleware/skills, not just edit existing files) and **strengthened** it (failure-trace feeding to the Diagnoser; capability-add hints).
- A **small-scale head-to-head ran** (7 tasks, 3 held-out). Both optimizers **independently converged on the same edit — adding file tools.** Ours **gated** it (didn't help the judging tasks → rejected → held baseline); AHE **committed** it blind (still 0/4 on the pool).
- **Partial verdict (valid):** on the locked held-out 3, **ours = baseline = 0.667** (`fix-git`=1.0, `sqlite-db-truncate`=1.0, `overfull-hbox`=0; clean `n=1` re-score, env healthy). "Ours = baseline" because our gate **accepted 0 edits** (it rejected the file-tools edit that didn't help — the never-regress discipline).
- **AHE held-out number: NOT captured.** A first scoring was corrupted by transient eval-box degradation (concurrent-load timeouts on a disk-pressured box — `fix-git` 1.0→0.0); the box then recovered and a clean `n=1` AHE re-score was running but **was stopped before completing** (per request to wrap up). So there is **no valid AHE held-out pass-rate**, hence **no final head-to-head verdict**. We refuse to report the corrupted 0/3 as a win — it isn't one.
- **Next:** finish the (one-command) clean AHE held-out re-score for a preliminary signal, then run the full, diverse TB 2.0 head-to-head on an adequately-resourced machine, where the optimizer's edge can actually show.

---

## 1. Objective

Decide whether harness_studio's "stochastic harness optimization" is good enough relative to its inspirations (SkillOpt, AHE, meta-harness), then **run our method against AHE on TB 2.0 optimizing the same input harness, and beat it.**

## 2. What was done

### 2.1 Understanding (all four codebases)
- **harness_studio** — two-speed optimizer: inner loop (find failures → diagnose/blame → competing Strategist edits → shell → structural check → **noise-aware gate** → snapshot) + outer loop (deep-audit rewind + family-map meta-agent). The gate is the *only* mutation point and is constructed with a Benchmark, never a Backend (structural trust boundary).
- **SkillOpt** — the methodological ancestor (Reflect = LLM gradient, edit-budget = LR with cosine schedule, accept/reject = line search, step_buffer = momentum, slow_update/meta_skill = two-speed). harness_studio generalizes its single greedy-gated Markdown skill into typed multi-file parts + a **noise-aware** gate + deep-audit rewind.
- **AHE** — evolves the same 7 components via `evaluate→analyze→improve`, but **greedy single-lineage, always-commits, no automatic rollback** (a regression poisons the lineage until the LLM happens to revert). Paper claim 69.7%→77.0% is **unreproduced prose** (its own work-report says full scale is infeasible on a laptop).
- **meta-harness** — Stanford IRIS TB2 reference (a different KIRA/Terminus2 harness on Runloop). Informative, but **not** the harness AHE optimizes.

### 2.2 Judgment — is harness_studio good enough?
**Core optimizer: yes — a strict conceptual superset of both inspirations**, faithfully implemented and (now) 88 tests green. Versus SkillOpt it adds typed multi-file edits, competing strategies, a noise-aware gate, and deep-audit rewind. Versus AHE it adds an elitist *never-regress* gate, which is the lever to beat AHE's greedy loop.
**But three real gaps surfaced when pointing it at the actual AHE target — all now closed in this branch:**
1. `KiraBenchmark` targeted the *wrong* harness (meta-harness KIRA/opus, not AHE's nexau/gpt-5.4) → built `NexauBenchmark`.
2. The optimizer could only *edit existing files*, while AHE *adds* tools → added a **directory-aware part map** so it can add capabilities (only the frozen `llm_config` stays off-limits).
3. The Strategist diagnosed from task *descriptions* only, while AHE reads failure *traces* → added **trace-feeding**.

### 2.3 Integration built
- `studio/benchmark/nexau.py` — `NexauBenchmark`: drives AHE's exact harbor/nexau path, resolves the 99 locally-cached TB2 tasks, threads `run_idx` for real wobble, captures failure-trace excerpts in-memory.
- `examples/` drivers — `run_nexau_tb2.py` (our arm), `tb2_ahe_arm.py` (AHE arm: generate config + run evolve.py + extract best), `tb2_config.py` (locked task split, env-overridable), `tb2_score.py`, `tb2_compare.py`, `_calibrate_nexau.py`, `TB2_HEADTOHEAD.md` (runbook).
- Fairness: identical input harness, actor model (gpt-5.4, env-locked so no edit can change it), harbor scorer, and optimization pool; locked held-out pile; the Strategist skill forbids touching `llm_config`.

### 2.4 Strengthening (the "tweak our method" work)
- **Trace-feeding** — `Benchmark.last_trace()`; the nexau adapter extracts the verifier failure output (`test-stdout.txt`) + the agent's last trajectory messages and feeds them to the Diagnoser. Degrades gracefully to `""`.
- **Capability-add hints** — the Strategist is explicitly prompted to add a missing tool/middleware and register it (AHE's main gain source), now reachable via the directory-aware part map.
- **Disk-bloat hardening** — the adapter deletes each harbor jobs dir after extracting traces (prevents unbounded disk growth over a long run; an earlier version of this fix had a missing `import shutil` — now fixed with a regression test that exercises the full `run()` path).

### 2.5 Validation
- **Calibration:** `fix-git` = 1.0 in 2.4 min via our adapter — proves the harbor invocation + reward parsing end-to-end.
- **Full-loop smoke:** one real round drove `Runner → Diagnoser → Strategist (real nexau edit) → shell → structural → real harbor gate (old vs new) → reject → snapshot`. Cost instrumentation correct.
- **88 unit tests** green (deterministic, no API).

### 2.6 The head-to-head run (small scale, feasibility-limited)
7 tasks, seed 0, held-out 3 (`overfull-hbox`, `sqlite-db-truncate`, `fix-git`); shared 4-task optimize pool.
- **Our arm** (2 rounds, trace-feeding on): baseline 0.667 → **0.667, 0 edits accepted.** It *proposed* real `read_file`/`write_file`/`edit_file` tools (160-line impls) + prompt edits, but the gate **rejected** them because the judging tasks fail on *reasoning correctness* (`regex-log`: agent's regex extracted wrong dates — it finished, wasn't a tool/timeout failure), which adding tools can't flip.
- **AHE arm** (2 iterations): **0/4 on the pool.** It evolved the *same* move — added file tools + "efficient structured workflows" guidance + memory — and **committed it blind** (no gate); still solved nothing.

## 3. Findings

1. **Our optimizer matches AHE's edit-discovery.** Both *independently* chose to add file tools to the bare agent. Our edit quality is competitive.
2. **The structural difference is real and observable.** Ours subjected the edit to an objective gate (rejected, held baseline); AHE committed it blind. This is harness_studio's core thesis, demonstrated mechanically.
3. **Why 0 uplift here:** the small task set's failures are *reasoning-limited*, not *harness-limited*. A 7-task slice rarely contains a genuinely harness-flippable task — which is exactly why AHE's own 7-point gain is spread across 89 tasks. The optimizer's value averages over a diverse set.
4. **The eval environment failed twice, invalidating the verdict:** (a) the local Docker box hit **disk 98% full**, so held-out scoring timed out (`fix-git` 1.0→0.0 — environmental, not the harness); (b) a `shutil` import bug (now fixed) crashed a follow-up probe. **No uncontaminated pass-rate comparison exists yet.** We will not present the corrupted numbers as a result.

## 4. Current status

| Piece | Status |
|---|---|
| Understand 4 codebases + judge harness_studio | ✅ |
| `NexauBenchmark` + drivers + fairness (dir-part-map) | ✅ committed |
| Strengthening (trace-feeding, capability hints, disk-fix) | ✅ committed |
| Calibration + full-loop smoke on real TB2 | ✅ |
| 88 unit tests | ✅ green |
| ours (= baseline) held-out 3 | ✅ **0.667** valid (`fix-git`✓ `sqlite-db-truncate`✓ `overfull-hbox`✗) |
| AHE-evolved held-out 3 | ❌ not captured — clean re-score stopped before completing |
| **Final ours-vs-AHE verdict** | ⏳ **incomplete** — needs the AHE held-out number (one command on a healthy box) |

## 5. Next plan

### 5.1 Immediate — get a valid number
- **Resolve the local box** (only if a quick local signal is wanted): free disk (`docker system prune -af` — only the *user* can authorize this; it was denied to the agent as a shared-resource deletion), then re-score baseline (= ours, since 0 edits accepted) and AHE-evolved on the held-out 3 **in one clean session at a consistent generous timeout, `n_concurrent=1`** (the prior failure was too-short timeout under concurrent load on a degraded box). A `tm=3` `fix-git` probe is running to distinguish *slow* from *dead*.
- **Likely local outcome:** a modest **never-regress** result (ours holds baseline; AHE may regress if its blindly-committed edit hurt an easy task) — validates the thesis but is not a strong "we optimized better" win.

### 5.2 Primary — the real verdict on an adequate machine
Run the full, diverse TB 2.0 head-to-head where the optimizer's edge can manifest. Turnkey from this branch:
```bash
git clone … && git checkout tb2-nexau-headtohead
export AHE_DIR=/abs/path/to/agentic-harness-engineering
# follow examples/TB2_HEADTOHEAD.md: uv sync AHE, harbor datasets download terminal-bench@2.0
python examples/tb2_score.py "$AHE_DIR/agents/code_agent_simple" --label baseline --k 3 -- ...
python examples/tb2_ahe_arm.py prepare --iterations 8 && (cd "$AHE_DIR" && uv run python evolve.py --config configs/experiments/exp-tb2-h2h.yaml)
python examples/run_nexau_tb2.py --rounds 6 --strategies 3 --proposer-model claude-opus-4-8
python examples/tb2_compare.py
```
Requirements: Docker with ≥24 GB RAM **and ample free disk**, gpt-5.4 creds, the `claude` CLI.

### 5.3 The levers to actually win (in priority order)
1. **Scale + diversity** — only the full set contains enough harness-limited tasks for accepted edits to raise the score.
2. **Never-regress edge** — over 89 tasks, AHE's greedy commits will sometimes regress; our gate won't. Widen the judging/audit pools and raise gate `k` to exploit this.
3. **Stronger proposer** — `--proposer-model claude-opus-4-8` for AHE-class edit quality (default Tier-A is sonnet).
4. **Trace-feeding (shipped)** — already feeds verifier+trajectory to the Diagnoser; consider also passing it straight to the Strategist instruction.
5. **More rounds + meta-loop** — `rounds > segment_length` so the family-map meta-agent fires and escapes plateaus.

### 5.4 Risks / caveats to disclose in any final write-up
- AHE arm ran **ADB-off + explore-off** (no SERPER) and **evolve effort = high** (the only locally-validated AHE path) — re-enable for a maximal-fidelity AHE run.
- Binary per-task scores are noisy; report pass-rate at `k≥3` and cost-per-point, not single rollouts.
- "Same input harness" = AHE's `code_agent_simple`; the model is env-locked to gpt-5.4 for both arms.

## 6. Resume / artifacts

- **Branch:** `main` (everything merged; feature branch deleted — single branch). ~20 files, ~1.3k lines: `studio/benchmark/nexau.py`, the dir-aware part map (`studio/parts.py`, `studio/components/shell.py`), trace-feeding (`base.py`/`instrument.py`/`runner.py`/`diagnoser.py`/`nexau.py`), `examples/tb2_*`, tests. 88 unit tests green.
- **Runbook:** `examples/TB2_HEADTOHEAD.md`. **Memory:** `tb2-headtohead-experiment`, `tb2-feasibility-facts`.
- **To finish the local preliminary verdict (one gentle command — env is healthy):**
  ```bash
  export TB2_TASKS="fix-git,cobol-modernization,overfull-hbox,sqlite-db-truncate,regex-log,git-leak-recovery,extract-elf" TB2_SEED=0 TB2_FINAL=3 TB2_AUDIT=1 TB2_JUDGING=2
  AHE_WS="$AHE_DIR/experiments/2026-06-09__11-44-30__tb2-h2h/workspace"   # AHE-evolved harness (adds file tools + workflow guidance)
  python examples/tb2_score.py "$AHE_WS" --label ahe --timeout-multiplier 3 --n-concurrent 1 --out /tmp/tb2_ahe.json
  python examples/tb2_compare.py     # baseline=ours 0.667  vs  ahe=?  -> verdict
  ```
  Interpretation: AHE-best `< 0.667` ⇒ ours wins (never-regress — AHE's blind commit hurt an easy task); `= 0.667` ⇒ tie; `> 0.667` ⇒ AHE's edit generalized (tweak ours and re-run). A decisive "we optimized better" win needs the full diverse TB 2.0 (§5.2).
