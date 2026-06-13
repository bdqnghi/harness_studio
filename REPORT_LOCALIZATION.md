# Evidence-Grounded Context Localization

*2026-06-13*

## Why

SHO hill-climbs a harness on a benchmark: run failing tasks → diagnose → propose a
hypothesis → implement an edit → a noise-aware acceptance check accepts/rejects. Investigating
**why edits weren't generalizing** exposed a broken context-localization path — the
single most likely cause:

- **Capture was a blind tail.** The tau2 adapter kept `reward` + `reward_breakdown[:500]`
  + the *last 4 messages* of one trial. In a multi-turn dialogue the causal mistake is
  usually mid-trajectory, not in the last 4 turns, and the verifier's structured signal
  (which named checks failed) was flattened to a string slice and never used.
- **Evidence died at diagnosis.** The diagnoser summarized the traces into a cluster that
  blamed one of 7 part-types — a **no-op when the editable surface is a single file**
  (tau2's `policy.md`). No intra-file (span/rule) anchor.
- **The editor edited blind.** The strategist received only a one-line `root_cause` +
  blame — **never the transcript.** It violated the basic rule *"don't change code you
  haven't read"* at the level that matters: the failure itself.

We studied how Claude Code (`/home/nghibui/codes/claude-code`) localizes and adapted the
principles (not the code): read-before-act is mandatory; localization is a read-only pass
that returns a compact, **evidence-cited conclusion**; search returns `file:line` with
context; delegation scales to difficulty.

## What changed

A real localization stage now sits between *diagnose* and *implement*, and the evidence
**survives to the editor**. It is benchmark-agnostic and applied to **both** optimizer
paths (classic + tree). Off by default in config; drivers default it on.

```
benchmark.run → adapter builds TaskEvidence{signals, causal windows, transcript}
              → EvidenceStore (versioned per harness hash)
round:  diagnose → materialize evidence → localizer.localize → validate citations
        → editor receives localized targets + evidence windows → acceptance check
```

### New modules
- **`studio/components/evidence.py`** — `VerifierSignal` / `TraceWindow` / `TaskEvidence`,
  `EvidenceStore` (LRU by harness hash, `materialize()` → per-task corpus files),
  `select_windows()` (picks the turns the *failed checks* point at; tail fallback when
  nothing localizes), `to_flat_excerpt()` (back-compat), `evidence_from_trace()` (coding
  adapters).
- **`studio/components/localizer.py`** — `localize()` in **inline** (`prompt_json`),
  **agentic** (`run_explore`), or **auto** mode (multi-file / many-cluster → agentic).
  A deterministic **citation guard** drops any target whose `current_text` / evidence
  quotes are not verbatim substrings of what was actually read — the code-side proof of
  read-before-act, with no LLM in the trust path. Never raises (→ `[]`).

### Changed
- **`backends`** — new `run_explore` (read-only Explore loop + `submit_findings` validated
  against a schema) on gemini/mock; base default raises (callers fall back to inline).
  `grep` gained `-C` context lines + `head_limit`.
- **`schemas.LOCALIZATION`**; **`config.LoopConfig.localizer`** (`off|inline|agentic|auto`,
  default `off`).
- **`benchmark/tau2.py`** — `_build_evidence` decodes `reward_info`
  (`action_checks`/`nl_assertions`/`db_check`/`communicate_checks`) → signals → causal
  windows; keeps the **worst** failing trial; `last_evidence`.
- **`benchmark/{nexau,mini_swe}.py`** — populate the evidence store from test stdout +
  agent trajectory (`evidence_from_trace`); `last_evidence`. (Localization generalizes to
  them for free.)
- **`components/strategist.py`** + **`ideator.py`** — carry the localized targets + the
  failing-task evidence into the editor / ideation; `read_dirs=[evidence_dir]` lets the
  editor pull full transcripts. Absent localization ⇒ byte-identical prompt (locked by a
  test). **`SKILL.md`** got the read-before-edit rule.
- **`orchestrator.py`** — `_localize()` wired into `_round` and `_round_tree`; emits a
  `localization_done` progress event.
- **`examples/hillclimb.py`** — `--localizer` flag (default `auto`).

## Validation

- **260 tests green** (`.venv/bin/python -m pytest`), incl. new suites:
  `test_evidence.py`, `test_explore_backend.py`, `test_localizer.py`, and integration
  tests proving the editor receives the localized block on **both** paths
  (`test_tree_loop.py`), plus the degrade-to-identical invariant when localization is off.
- **Live end-to-end on tau2-airline** (cold-start, `--localizer auto`): clean
  `noise floor → practice (3 real failures) → diagnose → localize → propose → acceptance check → verdict`
  with **`localization_done n_targets=2` in agentic mode** — the `run_explore` loop read
  the real evidence + harness and produced 2 targets that passed the citation guard. (The
  k=1 / 1-round config is a pipeline check, not a signal run: verdict 0.500→0.500,
  detectable ≈ 0.51.)

## Not yet done (run-enablement)

The localization *logic* generalizes to any benchmark whose adapter sets `evidence_store`
+ `last_evidence`. *Running* on the others still needs:
- **mini-swe** — adapter + evidence ready; register a Target (seed = AHE mini-swe-agent
  harness; `mini_swe_part_map` exists) and provide AHE repo + harbor + docker images.
  Multi-file → exercises the agentic `run_explore` path.
- **browsecomp** — no adapter exists yet; needs a `Benchmark` + `ColdStartBrief` (the
  cold-start "react" template is the vehicle).
