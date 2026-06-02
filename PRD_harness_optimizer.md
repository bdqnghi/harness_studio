# PRD — Self-Evolving Harness Optimizer

**Status:** Draft v1
**Owner:** (you)
**One-line:** A system that automatically improves an AI agent's harness (its codebase — instructions, tools, middleware, skills, memory) by repeatedly proposing coordinated changes, testing them against a real benchmark, and keeping only what genuinely helps — while a slower meta-loop revises *how* changes are proposed so the search keeps escaping plateaus.

---

## 1. Background and motivation

### 1.1 The problem
An AI agent's performance on a benchmark is bottlenecked less by the model and more by its **harness** — the scaffolding around the model: system prompt, tool descriptions, tool implementations, middleware, skills, sub-agent configuration, and long-term memory. Today this scaffolding is hand-tuned by engineers through slow trial and error. We want to automate that tuning as a controlled optimization process.

### 1.2 Why existing approaches fall short
Three prior systems each solved part of this, and each has a gap we address:

- **Component-decomposition optimizers (AHE-style):** expose the harness as ~7 editable parts and run an edit→attribute loop. Strong decomposition, but re-evaluate the *whole* benchmark every iteration — too expensive to scale, and blind to which edits will *regress*.
- **Optimizer-discipline systems (SkillOpt-style):** treat one artifact (a skill doc) as trainable, with bounded edits, a held-out acceptance gate, a rejected-edit buffer, and fast/slow updates. Excellent stability discipline, but optimize only a *single* untyped artifact.
- **Meta-evolution systems (AEVO-style):** separate "generate candidates" from "revise the mechanism that generates candidates," letting a meta-agent edit the search procedure between segments. This is what escapes long-horizon plateaus, but was not combined with typed multi-component harnesses or noise-aware gating.

### 1.3 Our thesis
Combine all three: **SkillOpt's optimizer discipline, generalized from one artifact to a typed multi-component harness (AHE), driven by minibatch hill-climbing for efficiency, with an AEVO two-speed structure so the search revises its own rules.** The acceptance decision stays an objective, code-only test against the real benchmark — which is our structural advantage over systems (e.g. self-evaluated tournaments) that must trust an AI judge.

### 1.4 Non-goals
- Not training model weights. The model is frozen; only the harness changes.
- Not a general agent framework. It optimizes an *existing* harness on an *existing* benchmark.
- Not full population-based evolutionary search (too expensive); we do single-line hill-climbing with a meta-loop that revises the search rules.
- Not human-in-the-loop per round; humans configure and monitor, the loop runs autonomously.
- **Not an optimizer for opaque harnesses.** The system does not operate on a target that is a single opaque invocation (e.g. a bare `claude -p` call with no exposed scaffolding). It requires a target whose editable parts are real, readable files/regions (see §1.5).

### 1.5 Hard precondition — the target harness must be an open codebase
The target harness (the thing being optimized) **must be a concrete, open agent codebase whose seven editable parts are exposed as files or code regions** — for example, mini-swe-agent. This is a hard requirement, because the entire pipeline operates on those files: the Mapper labels them, the Strategist edits them, the structural check compiles and boots them. An opaque single-call agent (a bare `claude -p` with no editable scaffolding) cannot be optimized — there is nothing to open up and edit.

**Selection criterion for the target harness:** choose one that is *structurally clean and conventional*. The cleaner the codebase's structure, the more reliably the Mapper labels it, and the Mapper bounds the quality of everything downstream. A messy or idiosyncratic target degrades the whole system at its root. mini-swe-agent is the natural starting pick: minimal, and its parts map cleanly onto the seven types.

> **Clarifying the "use `claude -p`" simplification.** Early in design we said "to save time, use `claude -p` instead of building a harness from scratch." That is correct *for our optimizer's AI helpers*, but it must not be read as "the target harness is `claude -p`." The two are different roles:
>
> | Role | Is it `claude -p`? | Why |
> |---|---|---|
> | **Our optimizer's AI helpers** (Mapper, Diagnoser, Strategist, Reviewer, Ranker, Meta-agent) | **Yes** — `claude -p` / Codex, swappable | They reason and suggest; we build prompts, not models. (AEVO's meta-agent is likewise "any coding-capable agent such as Claude Code or Codex.") |
> | **The frozen actor's model** (what runs benchmark tasks inside the harness) | **Swappable** — may itself be a CLI agent or an API model | Frozen during optimization; we never edit the model. |
> | **The target harness** (the thing optimized) | **No** — a real open codebase | It is the object we edit. Its parts must be readable/editable/compilable, or there is nothing to optimize. |
>
> The target harness *may call* `claude -p` (or any model) as its underlying engine — that is the frozen actor, and that is fine. What is not fine is the target *being* an opaque `claude -p` with no editable scaffolding. We optimize the **wrapper** (prompt, tools, middleware, skills) around a frozen model; the wrapper must be open. So "don't build the harness from scratch" is still true — its correct reading is **"use an existing open-source harness as the target,"** not "use `claude -p` as the target." The time-saving is real: we build no model and no harness, and our six AI helpers are all prompt-driven `claude -p`-style calls.

### 1.6 Which AI helper uses `claude -p` how — two tiers

Our six AI helpers are all `claude -p` / Codex-style calls, but they are **not the same kind of call.** Prior work draws a sharp line we adopt: a *proposer/editor* that must reason over a large, growing history of prior code and traces should be a **filesystem-navigating coding agent** (it inspects history via `grep`/`cat` and edits files through tool use), not a raw single-prompt call. Meta-Harness states this directly: the proposer is Claude Code (Opus-4.6), chosen as a *coding agent rather than a raw LLM* "because the amount of experience quickly exceeds context limits, so the proposer must decide what to inspect and validate edits through direct interaction with the codebase" — it reads a median of ~82 history files per iteration. AHE likewise implements its editor as a coding-agent "meta-agent" with access to prior code and distilled traces, not a fixed prompt. So we split our helpers into two tiers:

| Tier | Helpers | `claude -p` mode | Why this mode |
|---|---|---|---|
| **A — filesystem-navigating coding agent** (tool access; reads history via grep/cat; edits files) | **Strategist** (§5.3), **Meta-agent** (§5.10) | `claude -p` / `codex` run as a coding agent with a workspace + a minimal skill telling it where to read/write and what it may/may not modify | Their inputs (prior harness code, execution traces, the family map, the segment's accumulated evidence) exceed any fixed prompt. They must *selectively inspect* history and *edit files*, exactly Meta-Harness's argument. The Strategist edits the harness's parts; the Meta-agent inspects the segment's evidence on disk and rewrites the family map / Strategist skill. |
| **B — plain structured-output prompt call** (bounded input → JSON; no filesystem navigation) | **Mapper** (§5.0a), **Diagnoser** (§5.2), **Reviewer** (§5.4), **Ranker** (§5.5) | `claude -p` given a bounded prompt, returns JSON | Each consumes a bounded input (the codebase listing once; this round's failures; the round's proposed strategies) and emits a structured result. They don't need to roam history, so a plain call is cheaper and simpler. |

**Concretely, in the loop, the step that "provides results" by editing the actual harness files is the Strategist (Tier A).** It is the `claude -p` coding-agent invocation that reads the current harness + family map + relevant prior traces from the workspace and writes the candidate edits to the harness files — mirroring Meta-Harness's proposer. The **Meta-agent** (Tier A) is the other coding-agent invocation: once per segment it reads the accumulated evidence on disk and edits the *mechanism* files (family map, Strategist skill), never the harness or the gate. Everything else (Mapper, Diagnoser, Reviewer, Ranker) is a Tier-B bounded prompt call.

**Skill-guided, like Meta-Harness.** Each Tier-A agent is steered by a **minimal skill file** (per Meta-Harness's strongest practical lesson: "write a good skill") that specifies its workspace layout, what it may read, what files it may and may not modify, and its output contract — but leaves its *diagnosis and proposal reasoning* free. For the Strategist, the skill forbids touching do-not-touch files and the gate, and pins the per-part edit budgets. For the Meta-agent, the skill restricts edits to the mechanism files and forbids any contact with the evaluator/candidates (the AEVO protection rule).

---

## 2. Core concepts and glossary

| Term | Definition |
|---|---|
| **Harness** | The agent's editable codebase — an **open** codebase whose parts are real files/regions (see §1.5). The thing being optimized. The harness wraps a *frozen* model; we edit the wrapper, never the model. |
| **Actor** | The frozen model the harness drives to run benchmark tasks. Swappable; may itself be a CLI agent or an API model. Never edited during optimization. |
| **Editable part** | One of the labeled components that can be changed: instructions, tool descriptions, tool code, middleware, skills, sub-agent config, memory. Everything else is "do-not-touch." |
| **Strategy** | One complete, internally-coordinated proposal for fixing a round's failures — possibly touching several parts at once. The unit that competes. |
| **Strategy family** | A *class* of strategy (e.g. "tool-code timeout patches"), the grain at which lessons generalize. |
| **The wobble** | How much the benchmark score varies on its own (same harness, re-run). The noise floor. |
| **The gate** | The protected, code-only test that runs the harness old-way vs new-way and decides keep/reject. The only thing that changes the harness. |
| **Inner loop** | The fast per-round cycle: find failures → propose strategies → test → keep winner. |
| **Outer loop / segment** | The slow cycle (every K rounds): a meta-agent revises *how* strategies are generated. |
| **Strategy-family map** | The durable file the inner loop reads every round and the outer loop rewrites every segment. The interface between the two loops. |
| **Practice tasks / Judging set / Big audit set / Final exam** | The four disjoint task piles (see §6). |

---

## 3. Design principles (non-negotiable)

1. **Helpers suggest; code decides.** AI components only ever propose. Every keep/reject decision is plain arithmetic on real scores. Rationale: the component proposing a change is biased toward liking it; objective scoring is the only trustworthy referee.
2. **The gate is protected and external.** Neither the strategy-proposing AI nor the meta-agent can read the evaluator's internals, see hidden test artifacts, or write scores. Rationale: prior work showed agents with evaluator access reward-hack.
3. **One change at a time, attributable.** Strategies are kept small; the meta-agent makes exactly one mechanism edit per segment. Rationale: if many things change at once, you can't tell what helped.
4. **Spend the expensive resource (task runs) as little as possible.** Every cheap filter (does-it-run check, review, ranking) exists to keep the expensive gate from seeing bad candidates.
5. **Two speeds.** Generating strategies is fast and frequent; revising the rules that generate them is slow and periodic. Rationale: revising a *rule* needs a *pattern*, which only emerges across several rounds.
6. **Memory is compressed, not replayed.** The system never re-reads the full round-by-round transcript; it relies on the current harness (= all successes, embodied) plus the family map (= lessons) plus the avoid-list.

---

## 4. System architecture

### 4.1 Two loops joined by a shared file

```
SETUP (once)
   │
   ▼
OUTER LOOP  ── every K rounds ──────────────────────────────┐
   Meta-agent reads the segment's evidence, makes ONE edit   │
   to the strategy-family map (or the Strategist's rules),   │
   sets the run plan for the next segment.                   │
        │ writes                              ▲ evidence      │
        ▼                                     │               │
   ┌─ STRATEGY-FAMILY MAP (shared file) ─┐    │               │
   │  works / falsified / pivot / open   │    │               │
   └─────────────────────────────────────┘    │               │
        │ read every round                     │               │
        ▼                                     │               │
   INNER LOOP ── every round ─────────────────┘               │
   find failures → diagnose+blame → Strategist proposes        │
   (reads map) → review → rank → shell → does-it-run →         │
   GATE (protected, code) → snapshot                           │
                                                               │
   DEEP AUDIT at segment end feeds traps to the Meta-agent ────┘
```

### 4.2 Component roster

| # | Component | AI or code | Loop | Responsibility |
|---|---|---|---|---|
| 0a | Mapper | AI (Tier B) | setup | Label which files are which editable part. |
| 0b | Wobble measurement | code | setup | Measure the noise floor. |
| 0c | Task splitter | code | setup | Partition tasks into four piles. |
| 0d | Map initializer | code | setup | Create the empty strategy-family map. |
| 1 | Runner | code | inner | Run the harness on practice tasks; collect failures. |
| 2 | Diagnoser | AI (Tier B) | inner | Group failures, name causes, blame a part. |
| 3 | Strategist | AI (Tier A — coding agent) | inner | Propose competing strategies, guided by the map; writes edits to harness files. |
| 4 | Reviewer | AI (Tier B) | inner | Prune incoherent / known-dead strategies. |
| 5 | Ranker | AI (Tier B) | inner | Order survivors for testing (pre-filter only). |
| 6 | Code shell | code | inner | Enforce per-part edit limits, references, well-formedness. |
| 7 | Structural check | code | inner | Compile/load/boot — free, no task runs. |
| 8 | Gate (referee) | code | inner | Real test; keep iff gain beats the wobble. The only thing that changes the harness. |
| 9 | Snapshotter | code | inner | Save a rewind point each round. |
| 10 | Meta-agent | AI (Tier A — coding agent) | outer | Edit the family map / Strategist rules. Never touches the gate. |
| 11 | Deep auditor | code (+small AI) | outer | Wide-set check, rewind regressions, surface traps. |
| — | Orchestrator | code | both | Drives the loops, holds state, validates all AI outputs. |

---

## 5. Detailed component specifications

### 5.0a Mapper (AI, setup, one-shot)
- **Purpose:** define the optimization search space by labeling the harness.
- **Input:** the codebase (file tree, key files, README, entry points).
- **Output (JSON):** `{ part_type: [file paths or code regions] | "absent", ..., do_not_touch: [...] }` for the seven part types.
- **Behavior:** for each of the seven part types, locate the files/regions that implement it in *this* codebase, or mark absent. Everything unmapped → do-not-touch.
- **Re-run policy:** re-run at segment boundaries (codebase changes as edits land); newly-created unmapped files stay do-not-touch until re-mapped.
- **Failure mode to monitor:** misclassification → fixers edit useless code or can't reach the real problem. Track via the health signals (§7).

### 5.0b Wobble measurement (code, setup)
- **Purpose:** establish the noise floor used by the gate.
- **Behavior:** run the unchanged harness on a small fixed set of tasks N times (N small, e.g. 3–5); compute the spread of the aggregate score. Lock the value.
- **Output:** a scalar `wobble` (and optionally per-task variance).
- **Re-calibration:** periodically, or when fast-gate vs deep-audit disagreement rises (signals the wobble estimate drifted).

### 5.0c Task splitter (code, setup)
- See §6 for the four piles and their roles.

### 5.0d Map initializer (code, setup)
- Create an empty strategy-family map with the four sections (§5.10.2).

### 5.1 Runner (code, inner)
- **Purpose:** gather fresh failures for the round.
- **Behavior:** sample a fresh batch of **practice tasks** (size tuned so ~5–10 failures surface; grow as the harness improves). Run the current harness **once per task**. Grade each.
- **Output:** per-task trajectory + pass/fail.
- **Note:** these scores *locate failures only*; they make no decisions. One run per task is sufficient (precision is reserved for the gate). Use 2 runs per task only if the environment is flaky enough that single runs show spurious failures.

### 5.2 Diagnoser (AI, inner)
- **Purpose:** turn raw failures into causes + blame.
- **Input:** the failed trajectories from the Runner.
- **Output (JSON):** `[{ pattern_id, description, root_cause, failing_task_ids, blamed_part | "unclear", confidence }]`.
- **Behavior:** cluster failures by mode, infer root cause, point at the responsible part. Routing rides on this same call (no separate router).

### 5.3 Strategist (AI — Tier-A filesystem-navigating coding agent, inner) — the core proposer
- **`claude -p` mode:** run as a coding agent (`claude -p` / `codex`) with a workspace and a minimal skill (see §1.6). It reads the current harness files, the family map, and relevant prior traces from the workspace via grep/cat, and **writes its edits directly to the harness files** — the proposer pattern from Meta-Harness. This is the step that "provides results" by mutating the actual harness.
- **Purpose:** propose several competing strategies to fix this round's failures.
- **Input:** this round's failure patterns (§5.2) + **the current strategy-family map** + the editable-parts map + the avoid-list.
- **Output (JSON):** `[{ strategy_id, edits: [{ part, diff, intent }], rationale, family_label }]` — several strategies, each a complete coordinated plan.
- **Behavior / constraints:**
  - Each strategy may touch multiple parts but is internally coordinated (one mind designed it).
  - Each strategy as small as it can be while still addressing the failures.
  - Must respect the map: do not propose families listed under "falsified / do-not-repeat"; prefer families under "works"; bias toward "pivot toward" directions.
  - Tag each strategy with its `family_label` so the map and the meta-agent can reason about families.
- **Why one agent, not N parallel per-part fixers:** a single agent seeing all parts proposes *coordinated* multi-part fixes and avoids redundancy; separate per-part fixers would propose blind to each other and only discover clashes downstream.

### 5.4 Reviewer (AI, inner)
- **Purpose:** prune obviously bad strategies before any testing.
- **Input:** all strategies from the Strategist + the map's do-not-repeat list.
- **Output (JSON):** `{ keep: [strategy_ids], drop: [{ strategy_id, reason }] }`.
- **Behavior:** drop incoherent, implausible, or known-dead strategies. Reviews *whole strategies*, not fragments. Does not rank.

### 5.5 Ranker (AI, inner)
- **Purpose:** decide testing order so the most promising goes first.
- **Input:** surviving strategies.
- **Output (JSON):** ordered list of strategy_ids, best-guess first.
- **Critical constraint:** this is a **pre-filter, not a decision**. A mis-ranked strategy just gets tested first and rejected; the gate still decides. Ranking affects efficiency, never correctness.

### 5.6 Code shell (code, inner)
- **Purpose:** enforce the hard invariants the AI cannot be trusted with.
- **Per strategy:** enforce the per-part edit budget (clamp; overflow → avoid-list buffer); scan for broken references (one edit removes a name another needs → drop the dependent); validate well-formed diff touching only allowed parts.
- **Output:** validated strategies (in rank order) ready for the structural check.

### 5.7 Structural check (code, inner) — the free pre-gate
- **Purpose:** discard strategies that don't even run, before spending task runs.
- **Behavior:** apply the top strategy to a throwaway copy; check compiles? tools load? harness boots? Per-part first, then whole-harness smoke test. **Runs no benchmark tasks → free.**
- **On failure:** drop the broken strategy, record the exact error to the avoid-list; optionally give the Strategist ONE repair attempt with the error in context; else fall to the next-ranked strategy. If all fail, end the round (errors carried forward).

### 5.8 Gate (code, inner) — the referee, the only mutation point
- **Purpose:** decide whether a strategy genuinely improves the harness, and apply it if so.
- **Input:** harness old-way vs new-way (top surviving strategy applied), run on the **judging set** (stable within a segment; see §6), one run per task first.
- **Decision rule (three-way, noise-aware):**
  - `gain` = mean over judging tasks of `[score(new, task) − score(old, task)]` (paired).
  - **Clearly better** (gain ≫ wobble) → **accept**: the harness becomes the new version.
  - **Clearly not better** (gain ≤ 0) → **reject**; record outcome (note if it *regressed* — strong signal).
  - **Borderline** (0 < gain ≤ wobble band) → **re-run on additional close-call tasks**, capped (~5 extra). Clears the band → accept; still in-band at cap → reject.
- **On reject:** fall through to the next-ranked strategy (no regeneration) until one passes or the list is exhausted (then round ends).
- **Output:** updated harness (accepted) or unchanged (all rejected), plus every outcome written to the segment evidence record.
- **Protection:** runs in isolation; no AI component can read its internals, see hidden artifacts, or write scores.

### 5.9 Snapshotter (code, inner)
- **Purpose:** cheap rewind points.
- **Behavior:** the harness is text; save a full copy every round, tagged with round number and current score.

### 5.10 Meta-agent (AI — Tier-A filesystem-navigating coding agent, outer) — revises the search rules
- **`claude -p` mode:** run as a coding agent (`claude -p` / `codex`) with a minimal skill (see §1.6). It inspects the segment's accumulated evidence on disk (prior candidates' code, traces, scores — the AEVO/Meta-Harness history pattern) and edits only the *mechanism* files (family map, Strategist skill). Its skill forbids any contact with the evaluator or candidate scores.
- **Purpose:** edit *how strategies get generated* so the search escapes plateaus. Does **not** propose strategies and **cannot** touch the gate.
- **Trigger:** once per segment (every K rounds), at the boundary, after the deep audit (§5.11) has produced trap evidence.
- **Input:** the segment's accumulated evidence record — which families were accepted, which rejected and how (no-help vs regress), which passed the fast gate but failed the deep audit (traps), repetition/churn patterns.
- **Action (exactly ONE per segment, for attributability):**
  - **Update the family map** (most common): promote a confirmed-working family to "works/prefer"; add a confirmed-dead family to "falsified/do-not-repeat" with reason; add a "pivot toward" directive when a class has stalled.
  - **Edit the Strategist's instructions:** e.g. "you over-anchor on tool-code; diversify across parts," or change how failure evidence is formatted to it.
  - **Adjust the run plan:** lengthen the next segment if progress is steady, shorten it if churning.
- **Output:** updated family map + (optional) updated Strategist instructions + next-segment run plan.
- **Hard rule:** may edit only the mechanism files (map, Strategist prompt, run plan). May not edit `candidates/`, the evaluator, or scores.

#### 5.10.2 The strategy-family map (format)
A persistent file with four sections:
- **Works (prefer):** families confirmed to help, one-line why.
- **Falsified (do not repeat):** families that failed the gate *or* the deep audit, with reason. (Deep-audit traps are the highest-value entries.)
- **Pivot toward:** explicit next-direction directives for stalled classes.
- **Open / untried:** families not yet explored (the frontier).

Grain is **families** (classes of approach), not individual edits — "stop trying tool-code for timeouts" generalizes; "don't repeat edit #47" does not.

### 5.11 Deep auditor (code + small AI, outer)
- **Purpose:** catch what the fast gate cannot, and manufacture the trap evidence the meta-agent needs.
- **Trigger:** segment boundary, before the meta-agent acts.
- **Behavior:** run the current harness on the **big audit set** (large, mostly-untouched pile).
  - Genuinely better → save as new best version.
  - Secretly worse (lucky judging-set wins, or changes that fight each other) → **rewind** to the last good snapshot.
  - Identify families that passed the fast gate but failed here → tag "trap," feed to the meta-agent as falsified.
- **Health checks (also here):** see §7.

### 5.12 Orchestrator (code, both loops)
- **Purpose:** the deterministic spine.
- **Responsibilities:** drive the inner and outer loops; hold all state (current harness, family map, avoid-list, snapshots, evidence record, health counters); **validate every AI output** (schema-valid JSON; reject edits to wrong parts; retry-once-or-skip on malformed output); enforce that no AI component crosses the gate boundary.

---

## 6. Data: the four task piles

| Pile | Role | Sampling | Used by |
|---|---|---|---|
| **Practice tasks** | Find failures to learn from | Fresh-random each round | Runner (§5.1) |
| **Judging set** | Score whether a change helped | Stable within a segment, rotated between segments | Gate (§5.8) |
| **Big audit set** | Thorough generalization double-check | Large, mostly untouched; optionally rotated | Deep auditor (§5.11) |
| **Final exam** | The single honest final number | Locked, never touched until the very end | Final report only |

**Why separate:** prevents the harness from "cheating" by overfitting the exact tasks it's repeatedly scored on. The judging set is stable within a segment (for comparability and caching) but rotated between segments (to avoid overfitting it). The final exam guarantees an honest headline number.

---

## 7. Health monitoring and failure handling

The Orchestrator tracks these signals; each crossing a threshold triggers a defined response:

| Signal | Meaning | Response |
|---|---|---|
| Consecutive empty rounds (everything dropped at shell/structural check) | Strategist producing junk / lacks context | Feed the Strategist more context (the relevant part's code, an example valid edit); if persists, re-map or stop. |
| Long gate-rejection streak | Search stuck | Trigger an early meta-agent intervention (pivot directive). |
| Fast-gate vs deep-audit disagreement rising | Wobble estimate drifted, or judging set being overfit | Re-measure the wobble; shorten the judging-set rotation. |
| High fraction of "unclear"/unroutable failures | Diagnoser can't blame a part, or the cause is in do-not-touch code | Surface to humans; may indicate the editable-part map is too narrow. |
| Repeated near-identical strategies | Strategist anchoring | Meta-agent issues a diversify/pivot directive. |
| Reward-hacking attempt detected (impossible scores, evaluator-boundary probing) | Critical | Halt; the gate's isolation should prevent this, but flag and stop. |

---

## 8. The rollout / cost policy (efficiency)

- **One rollout per sample by default**, everywhere. Task runs are the expensive resource.
- **Wobble calibration:** a few repeated runs once, at setup, then locked (re-calibrate occasionally).
- **Gate:** one run per judging task first; add runs only on *borderline* decisions, only on the contested tasks, capped (~5).
- **Deep audit:** stability from *breadth* (large set, run once each), not from per-task repeats.
- **Caching:** hash candidate harnesses; reuse scores within a segment (judging set is stable, so cache hits are possible).
- **The meta-agent is expensive** (deep deliberation, ~3× a normal round in prior work); running it once per segment amortizes that cost. For short runs, the family map can be updated by cheap rules instead of a full meta-agent, escalating to the meta-agent only on observed plateaus.

---

## 9. Success metrics

### 9.1 Primary
- **Final-exam score uplift** vs. the unmodified harness baseline, on the locked final-exam pile.
- **Per-family score deltas** (not just aggregate): the system should improve hard/minority failure families, not only pad easy common ones. A rising aggregate that hides neglected hard families is a failure.

### 9.2 Efficiency
- **Task runs per point of improvement** (the cost-per-point metric). Compare against an AHE-style full-evaluation baseline and a SkillOpt-style single-artifact baseline.
- **Rounds-to-plateau** and **whether the meta-loop produces post-plateau jumps** (the AEVO signature — improvement continuing after a fixed-rule baseline flattens).

### 9.3 Health / trust
- **Fast-gate vs deep-audit agreement rate** (high = the cheap check is honest).
- **Reward-hacking incidents** (target: zero, enforced by gate isolation).
- **Strategy acceptance rate** and **trap rate** (passed fast, failed deep) over time.

---

## 10. Build phases / milestones

**Phase 0 — Single-line baseline (no meta-loop, no typing).**
Implement the inner loop only, on one untyped editable artifact, with the noise-aware gate. This is essentially SkillOpt. Validates the gate, the wobble calibration, the buffer, and the snapshot/rewind. Establishes the baseline to beat.

**Phase 1 — Typed multi-part harness.**
Add the Mapper, the seven editable parts, per-part edit budgets, the structural pre-gate (compile/load/boot). Validates that typing + the free structural check improve efficiency and stability over Phase 0. This is the AHE generalization.

**Phase 2 — Strategy unit + diagnose/blame.**
Add the Diagnoser (blame), the Strategist (competing whole-strategy proposals), Reviewer, Ranker. Replace per-edit handling with strategy-as-bundle. Validates coordinated multi-part fixes.

**Phase 3 — The two-speed meta-loop.**
Add the strategy-family map, the Meta-agent, the deep audit, the segment structure. Validates that mechanism-editing escapes plateaus (the AEVO contribution). This is the full system.

**Phase 4 — Hardening.**
Health monitoring, reward-hack defenses, caching, cost instrumentation, ablations for the success metrics.

---

## 11. Open questions / decisions to make

1. **Which target harness?** (decide first) — must be an open codebase with cleanly-exposed parts (§1.5). mini-swe-agent is the recommended starting pick (minimal, conventional structure → reliable Mapper). The choice bounds Mapper quality and therefore everything downstream, so pick for *structural cleanliness*, not feature richness.
2. **Segment length K** — fixed, or meta-agent-chosen per run plan? (Recommendation: meta-agent-chosen, bounded.)
3. **Test top-1 or top-few strategies per round?** — top-1-with-fallthrough is cheapest; top-few gives the *real test* (not the ranker) the final say among rivals, at higher cost. Decide based on measured cost-per-task-run.
4. **Per-part edit budgets** — should they be fixed, or adapt to each part's observed edit-outcome variance? (Adaptive is more principled but adds machinery; start fixed.)
5. **Meta-agent vs. rule-based map updates** — full AI meta-agent from the start, or cheap rule-based map updates with the meta-agent as an escalation? (Decide on run length / budget.)
6. **Repair attempts in the structural check** — allow one, or zero? (One salvages typo'd-but-good edits cheaply; cap strictly at one.)

---

## 12. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Trace-based blame is wrong → Strategist edits the wrong part | Medium | The gate rejects bad strategies anyway; track blame hit-rate; re-route persistent failures. |
| Editable-part map (Mapper) misclassifies the codebase | High | Bounds the whole search; validate on a sample; surface via "unclear" failure rate. |
| Judging-set overfitting over a long run | Medium | Rotate the judging set between segments; the deep audit + final exam are the backstops. |
| Meta-agent cost dominates | Medium | Once-per-segment cadence; rule-based fallback for short runs. |
| Reward-hacking via evaluator access | High | Hard gate isolation; no AI component touches the evaluator (validated as critical in prior work). |
| Non-stationary wobble (edits change variance) | Low–Med | Periodic re-calibration; watch fast-vs-deep agreement. |

---

## Appendix A — One-round walkthrough (concrete)

1. Runner samples 18 fresh practice tasks, runs each once. 7 fail.
2. Diagnoser: "5 timeouts (blame: tool code), 2 wrong-format (blame: instructions)."
3. Strategist reads the family map — which says *"falsified: tool-code timeout patches (trap, failed deep audit twice); pivot toward: middleware for timeouts."* It proposes: **Strategy A** = {middleware: add a timeout wrapper} + {instructions: add a format rule}; **Strategy B** = {instructions: format rule only}; **Strategy C** = {skills: add a verification step}.
4. Reviewer drops nothing (all coherent, none on the do-not-repeat list — note it did *not* propose a tool-code timeout patch, because the map forbade it).
5. Ranker orders: A, B, C.
6. Shell clamps budgets, scans references — all clean.
7. Structural check on A: compiles, boots. Pass.
8. Gate: old harness 64% on judging set, new (A) 71%. Gain +7%, wobble ±2% → clearly better → **accept**. Harness is now version with A applied.
9. Snapshot saved.
10. ... rounds 2–10 proceed similarly under the same map ...
11. **Segment boundary:** deep audit runs A's family on the big audit set — middleware-timeout family holds up (not a trap). Meta-agent reads the segment: middleware-timeout worked 4×, format-rules worked 3×, one skills-family churned with no acceptance. It makes ONE edit: promotes "middleware timeout handling" to *works/prefer* and adds "skills-verification family: no traction, deprioritize." Sets next segment = 10 rounds. Inner loop resumes under the updated map.

---

## Appendix B — What is AI vs. code (the trust boundary)

- **AI (suggests, can be wrong, swappable model calls):** Mapper, Diagnoser, Strategist, Reviewer, Ranker, Meta-agent. **These are our optimizer's AI helpers — all implemented as prompt-driven `claude -p` / Codex-style calls. We build prompts, not models.** They come in two tiers (§1.6): **Tier A — filesystem-navigating coding agents** that read history via grep/cat and edit files (the **Strategist** and **Meta-agent**); **Tier B — plain structured-output prompt calls** (Mapper, Diagnoser, Reviewer, Ranker). The Tier-A split follows Meta-Harness (proposer = Claude Code, a coding agent not a raw LLM) and AHE (Evolve Agent = a workspace-constrained coding "meta-agent").
- **Code (decides/enforces, deterministic, the trust backbone):** Orchestrator, wobble calibration, task splitter, Runner, code shell, structural check, **the gate**, snapshotter, deep auditor (rewind logic), health monitor.
- **Neither AI-helper nor optimizer-code — the things being acted *on*:** the **target harness** (an open codebase we edit; never an opaque `claude -p`) and the **frozen actor model** it drives (swappable, never edited). See §1.5.
- **The line that never moves:** no AI component decides what to keep or touches the gate/evaluator. The gate is plain arithmetic on real benchmark scores, in isolation.
