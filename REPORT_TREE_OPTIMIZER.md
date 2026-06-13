# Progress Report — Option B: The Hypothesis-Tree Optimizer

**Date:** 2026-06-11 · **Branch:** `main` · **Status:** implemented and validated (201 tests green); live TB2 runs launching now on two API lanes (gemini-3.5-flash and gpt-5.4 in parallel).

This report explains, in plain language, everything that changed for "Option B" — the new tree-based optimizer — plus the two real-world bugs we caught during launch preparation and the experiment that is now starting.

---

## 1. Why we built B (the problem with the old loop)

The old optimizer ("classic", Option A) works like this every round:

1. Run the agent on a batch of tasks, collect failures.
2. An LLM diagnoses the failures.
3. **Two full coding-agent runs** each produce a complete candidate edit.
4. The gate measures each candidate (64+ benchmark rollouts per candidate) and keeps at most one.

A deep review (12-agent analysis of our code plus the Arbor, Self-Harness, AHE, and SkillOpt papers) found the expensive part is not just the measuring — it's the **forgetting**:

- **A rejection teaches nothing.** The only thing that happened on a rejected edit was a counter going up (`orchestrator.py:365`). The same dead idea could be proposed again next round and re-measured for another 64–128 rollouts.
- **Paid-for ideas were thrown away.** Every round generated 2 full candidates; the loser (or the never-tested one) was deleted. Next round, a fresh LLM call regenerated something similar from scratch.
- **The expensive step ran first.** Full coding-agent implementations were produced *before* any cheap filtering, so filtering only saved rollouts, never proposer cost.

The Arbor paper (arXiv 2606.11926) showed the missing ingredient is **memory**: its ablations found that keeping a persistent tree of tried ideas, and propagating the *lessons* from each test, mattered more than anything else they did (score dropped from 81.8% to 54.5% without lesson propagation).

## 2. What B is, in one paragraph

B keeps a small persistent **tree** on disk. The tree has two levels: **directions** (groups of failures that share a root mechanism, e.g. "agent drowns in long tool output") and **hypotheses** under each direction (concrete edit ideas, e.g. "cap tool output at 2,000 characters in middleware"). Each round, B picks the most promising direction using the evidence so far, takes an already-written hypothesis from that direction's queue (or writes a few new ones as cheap text — no coding agent involved), implements **only the chosen one** with a single coding-agent run, and measures it with the **exact same gate** as the classic loop. The verdict is then written back into the tree forever: proven-bad ideas become "do not re-propose" constraints, ideas killed by measurement noise stay retryable, and every test leaves behind a short written lesson that future ideas inherit.

## 3. How a round of B works, step by step

1. **Find failures.** Same as before: run the practice batch, collect failing tasks with their failure traces.
2. **Diagnose (upgraded).** The diagnoser now also reports a *failure signature* for each failure cluster: what the verifier observed, what the agent did wrong, and — importantly — whether the failure is **addressable** at all (a harness edit could plausibly fix it). Unfixable clusters (infrastructure flakes, raw model limits) are dropped before any money is spent on them. This idea comes from Self-Harness.
3. **Route to directions.** One cheap LLM call matches this round's failure clusters onto the existing direction nodes, or creates new directions. (Free-text signatures never match exactly across rounds, so an LLM does the matching; if it fails, every cluster just becomes a new direction — wasteful but never wrong.)
4. **Pick a direction.** Seeded Thompson sampling over each direction's track record (a Beta distribution per direction: accepted children push it up, falsified children push it down, noise-rejections push it down at half weight). Early on this explores everything; as evidence accumulates it concentrates on what works. Same seed → same choices → reproducible runs.
5. **Pick a hypothesis.** If the direction has pending (already-written, never-tested) hypotheses, take the first — **zero new LLM cost**. Otherwise one cheap text call writes 4 new hypotheses, each forced into Arbor's discipline: *Mechanism* (how the change works), *Hypothesis* (the concrete edit), *Observable* (the measurable effect predicted if it's right). The prompt includes the validated lessons from sibling ideas, the full "do not re-propose" list of falsified ideas, and the pending queue (to avoid duplicates).
6. **Implement exactly that hypothesis.** One coding-agent run with the hypothesis quoted verbatim as a fixed, non-negotiable contract. The agent implements it; it does not get to substitute its own idea — the competition between ideas already happened cheaply as text in step 5.
7. **Gate it.** Byte-identical gate code to the classic arm (we deliberately did not touch `gate.py` — verified by a zero-line diff).
8. **Remember the verdict.** Three outcomes:
   - **Accepted** → harness updated, node marked `tested_accepted`, a ≤200-word lesson is distilled onto the node, and the direction's summary is refreshed.
   - **Clearly worse than noise** → `falsified`. The idea is dead forever and its text joins the constraint list every future ideation sees.
   - **Rejected but within the noise band** → `rejected_noise`. The idea stays retryable (at most twice) — this distinction is ours, not Arbor's: Arbor ignores measurement noise; we cannot, because hard-banning an idea that *noise* killed would permanently lose good ideas.
9. **Segment boundaries** (every 2 rounds): the deep audit still runs. If it catches a secret regression, the harness is rewound **and the responsible nodes are falsified** — the trap becomes a permanent lesson. The old "meta-agent" escalation is gone in B: the tree pivots structurally (constraints + shifting posteriors) instead of an LLM editing a strategy file.

**What this saves:** rejections stop being pure waste (they become constraints and lessons), unexecuted ideas stop being regenerated (the frontier holds them), and the expensive coding-agent step runs once per round instead of twice — on an idea that already won a cheap text-level competition.

## 4. Everything that changed (file by file)

### New files
| File | What it is |
|---|---|
| `studio/components/idea_tree.py` | The tree: nodes, statuses, posteriors, Thompson selection, `classify_rejection` (falsified vs noise), atomic save to `idea_tree.json` + human-readable `tree.md` on every change (also the crash-resume state) |
| `studio/components/ideator.py` | The two cheap text calls: direction router + hypothesis ideation (with the constraint/lesson/queue context) |
| `studio/components/insight.py` | Lesson distillation: ≤200 words per tested node, plus direction summaries; failures degrade to empty strings, never to a failed round |
| `studio/observe.py` | `ProgressLog`: the append-only `progress.jsonl` event stream (see §6) |
| `examples/tb2_ab_compare.py` | The readout script: per-arm lift ± error, per-task differences, efficiency table, pre-registered verdict sentence |
| `REPORT_TREE_OPTIMIZER.md` | This report |

### Changed files
| File | Change |
|---|---|
| `studio/orchestrator.py` | New `_round_tree` / `_test_tree` / `_segment_boundary_tree` path selected by `LoopConfig.optimizer` (`"classic"` or `"tree"`); classic methods untouched apart from progress-event emission; full event instrumentation of both paths |
| `studio/config.py` | `LoopConfig.optimizer`, `LoopConfig.hypotheses_per_direction`, `Config.score_cache` |
| `studio/schemas.py` | DIAGNOSIS gains the signature triple (optional, default-filled); 3 new schemas for router/ideation/insight |
| `studio/components/diagnoser.py` | Asks for and default-fills the signature triple — shared by both arms, identical output |
| `studio/components/strategist.py` | New `implement_hypothesis()` (stage-2 implementer); existing functions untouched |
| `studio/benchmark/instrument.py` | Disk-backed, namespaced score cache (survives restarts; one shared file can serve k=1 gate and k=3 verdict benches without cross-contamination; disk content is re-validated by the reward-hack guard) |
| `studio/benchmark/nexau.py`, `mini_swe.py` | Failure traces now keyed by `(harness, task)` — a candidate's run can never be misattributed to the live harness (this was a real bug: the diagnoser could read a *rejected candidate's* trajectory and blame the wrong code) |
| `studio/state.py` | `progress.jsonl` / `health.log` paths; health signals persist to disk |
| `studio/backends/mock.py` | Records prompts so tests can assert what the LLMs were actually told |
| `examples/tb2_self_compare.py` | New flags: `--optimizer`, `--hypotheses`, `--score-cache`, `--calibrate-only`, `--baseline-out`, `--source-harness` |

### Tests (49 new; 201 total, all green)
- `tests/test_idea_tree.py` — persistence/resume, atomic writes, posterior math, falsified-vs-noise classification across all four gate-decision shapes, frontier ordering, deterministic selection.
- `tests/test_tree_helpers.py` — router reuse/create, constraint header injection, the implementer quoting the hypothesis verbatim, insight truncation.
- `tests/test_tree_loop.py` — full 3-round toy run: statuses land correctly; **frontier reuse proven by not scripting a second ideation response** (the mock errors if consulted); falsified text + sibling lessons demonstrably appear in the next ideation prompt; audit trap falsifies and rewinds; the classic arm never creates tree files and the tree arm never calls classic-only helpers.
- `tests/test_toy_ab_noise.py` — both arms, 5 random seeds under injected noise: the known-bad edit is never accepted by either arm; noiseless, both reach the toy optimum and the tree uses no more coding-agent calls than classic.
- `tests/test_observe.py`, `tests/test_ab_compare.py`, plus updated trace/instrument/protocol tests.

## 5. The two launch-day bugs the smoke tests caught (and why they mattered)

Before spending hours of compute we ran cheap one-task "smoke" probes. They caught two issues that would have invalidated the entire run **silently** — every task would have scored 0 and the optimizer would have spent the whole budget "optimizing" pure noise:

1. **The ARM/`--force-build` trap.** This machine is ARM (aarch64). When harbor is invoked *without* `--force-build`, it uses its prebuilt task images, which are x86-only — the container starts but agent setup crashes within seconds. My earlier "optimization" (skip `--force-build` for already-built tasks) was therefore wrong on this hardware: the first harbor call of a run would work and **every subsequent call would fail**. Reverted: every invocation forces a build (docker layer caching makes repeats cheap), with a test pinning the behavior and a comment explaining why.

2. **The missing `LLM_API_TYPE`.** Our model-general harness config selects its provider path via the environment variable `LLM_API_TYPE`. Harbor forwards only a fixed allowlist of variables into the task container — and that variable is not on the list. Result: the agent crashed on config load *inside* the container, harbor recorded a clean trial with reward 0, and nothing upstream saw an error. **Every smoke score of 0.0 looked like "hard task" but actually meant "agent never ran."** Fix without touching harbor's vendored code: per-lane copies of the source harness with the provider baked in (`artifacts/b_runs/src_gemini` → `api_type: gemini_rest`; `artifacts/b_runs/src_openai` → `api_type: openai_responses` plus the OpenAI base URL), wired in via the new `--source-harness` flag.

After the fixes we verified the agent **genuinely acts**: the gpt-5.4 probe produced a real 17-message working trajectory (100 KB trace, actual file edits). The gemini probe is finishing the same check as this report is written.

This is also a lesson now encoded in the watchdog: *calibration mean pass-rate of exactly 0 across 64 tasks = "agent is dead", abort immediately* — score 0 must never again be confused with "task is hard."

## 6. The experiment now starting

Per your instruction, we run **only B** (no classic arm) and judge it against the *baseline harness* on locked held-out tasks — the driver's built-in verdict already measures that.

**Two lanes in parallel** (separate API quotas, no rate-limit contention):

| Lane | Actor + proposer | Source harness | Workspace |
|---|---|---|---|
| 1 | gemini-3.5-flash | `artifacts/b_runs/src_gemini` | `artifacts/b_runs/gemini` |
| 2 | gpt-5.4 | `artifacts/b_runs/src_openai` | `artifacts/b_runs/openai` |

Both: `--optimizer tree --rounds 10 --segment-length 2 --seed 0 --n-concurrent 8`. Identical frozen split (32 held-in pool = judging, 32 regression, 16 audit, **25 locked test tasks** — 14 long-running ones included — that the optimizer never sees).

**Phases per lane:** baseline calibration (64 tasks × 3 rollouts ≈ 3–4 h) → noise-floor measurement → 10 tree rounds (≈ 1–2 h each) → final verdict (baseline vs optimized harness on the 25 locked tasks at 3 rollouts each, heavy tasks make this 4–6 h). Total ≈ 1–1.5 days per lane, running concurrently.

**What "B works" means, concretely:**
1. *Loop health:* rounds complete; some hypotheses get accepted; falsified ideas never reappear (re-proposal rate ~0); insights accumulate in `tree.md`; deep audits confirm accepted edits rather than rewinding them.
2. *The number:* the final `lift.json` per lane — mean per-task improvement on the 25 locked tasks ± standard error, against a detectable floor of ≈ 0.14. Positive lift beyond the error bar on both lanes = strong signal; positive but inside the bar = weak signal, judged together with loop health; zero accepts or rewound-everything = B needs algorithm changes (which we then make and re-run).

**Watchdog:** I wake periodically, parse each lane's `progress.jsonl` / `evidence.jsonl` / `tree.md`, and report: current phase, round number, accepts/rejections, rollout count, tree state, and any alarms. Hard abort criteria: noise floor > 0.25 (gate would be blind), calibration mean exactly 0 (dead agent), repeated harbor failures, or > 2,400 rollouts in one arm (runaway). You can always look yourself:

```bash
tail -f artifacts/b_runs/gemini/progress.jsonl   # live event stream
cat artifacts/b_runs/gemini/tree.md              # current hypothesis tree
tail -f artifacts/b_runs/gemini.log              # driver log (same for openai)
```

## 7. Current status & next steps

- Implementation: **done**, 201/201 tests green, classic path frozen and untouched.
- Smoke + agent-acting verification: **openai lane confirmed**; gemini lane probe finishing now.
- Launch: both lanes start the moment the gemini probe confirms (the watchdog does this automatically).
- First substantive report: after calibration (~4 h in) — baseline pass rate and measured noise per lane.
- If B underperforms: the tree artifacts (`idea_tree.json` with statuses, evidence, insights per node) are exactly the autopsy data we need to tweak the algorithm — that is the next loop of this project either way.
