# Plan — Editable Surface v2 (replace the 7-file-bucket Mapper)

## 0. Why (one paragraph)

Today the optimizer can only edit **whole files** sorted into **7 fixed types** discovered by an LLM that reads a file tree. This is shaped by AHE's tidy nexau harness and (a) doesn't generalize to real, code-heavy harnesses where the prompt is an f-string and tools are functions; (b) gives coarse, hard-to-attribute edits; and (c) **structurally can't reach the highest-leverage surfaces** — few-shot demonstrations, control-flow, and hyperparameters (`max_iterations`, retries) — which the literature says are where the wins live. The accuracy ceiling is set by *what's reachable to edit*; this plan raises it. It is **additive and backward-compatible**: the existing whole-file/7-type path keeps working and the 100 tests stay green; new power is opt-in.

## 1. Design principles

1. **Span, not file.** The unit of edit is a named, addressable *region* (a YAML key, a markdown section, a Python function/docstring/constant, or a whole file as the degenerate case).
2. **Tag, not fixed type.** Surface types are an open registry. The 7 stay as built-ins; we add `DEMONSTRATIONS`, `HYPERPARAMS`, `CONTROL_FLOW`, `RETRIEVAL`.
3. **Discover structurally, tag cheaply.** Parse structure deterministically (AST/YAML/markdown); use a small LLM call only to *tag* ambiguous spans — never to blindly bucket whole files.
4. **Adapters for real frameworks.** Ship adapters that know where components live in nexau / Claude Code / LangGraph / OpenAI Agents SDK; fall back to the generic structural extractor.
5. **The tag drives policy.** Surface tags feed both the family map (steering) *and* the new acceptance rule (additive vs behavioral — see §7), so this plan and the gate redesign reinforce each other.

## 2. Core data model (`studio/surface.py`, new)

```python
class SurfaceType(str, Enum):              # OPEN registry, not closed
    INSTRUCTIONS=...; TOOL_DESCRIPTIONS=...; TOOL_CODE=...; MIDDLEWARE=...
    SKILLS=...; SUBAGENTS=...; MEMORY=...          # the legacy 7 (unchanged values)
    DEMONSTRATIONS="demonstrations"                # NEW: few-shot examples
    HYPERPARAMS="hyperparams"                      # NEW: max_iterations, retries, temperature(bounded)
    CONTROL_FLOW="control_flow"                    # NEW: agent loop / routing / stop conditions
    RETRIEVAL="retrieval"                          # NEW: what goes into context
    # plus SurfaceType.register("custom_tag") for unknown frameworks

@dataclass(frozen=True)
class Locator:
    """How to find a span inside a file. kind ∈ {whole_file, yaml_path, md_section, py_node, line_range, json_path}."""
    kind: str
    ref: str            # e.g. "tools[3].description" | "## Strategy" | "func:read_file/docstring" | "12-40"

@dataclass
class EditableSpan:
    span_id: str        # STABLE id, anchored by name/key NOT line number: "code_agent.yaml#hyperparams.max_iterations"
    path: str           # file it lives in
    locator: Locator
    surface: SurfaceType
    editable: bool = True
    meta: dict = field(default_factory=dict)   # HYPERPARAMS: {type:int, min:1, max:300}; DEMONSTRATIONS: {schema:...}
    def read(self, harness) -> str: ...         # current content of just this span
    def write(self, harness, text) -> None: ... # replace just this span in the file (structure-preserving)

@dataclass
class SurfaceMap:                               # SUPERSET of today's PartMap
    spans: list[EditableSpan]
    do_not_touch: list[str]                     # files with no editable span
    # views used by the rest of the system:
    def spans_for(self, s: SurfaceType) -> list[EditableSpan]: ...
    def span_of(self, span_id) -> EditableSpan | None: ...
    def editable_in(self, path, byte_range) -> bool: ...   # is this edit inside an editable span?
    def surface_of_change(self, path, byte_range) -> SurfaceType | None: ...
    # BACKWARD COMPAT (keeps all legacy code working unchanged):
    def to_partmap(self) -> PartMap: ...        # collapse spans -> {legacy7 type: [files]}; new tags map to nearest legacy or are dropped for legacy callers
    @classmethod
    def from_partmap(cls, pm) -> "SurfaceMap": ...  # each mapped file/dir -> one whole_file span
```

**Span identity must survive edits.** IDs anchor on names/keys (`tools[name=read_file].description`, `## Safety rules`), not line numbers, so an accepted edit doesn't invalidate the map. Re-extract after each accepted edit (cheap, same cadence the Mapper re-runs today).

## 3. Discovery — extractors + adapters (`studio/extract/`)

A pluggable pipeline: `extract_surface(harness, framework=None) -> SurfaceMap`.

**Structural extractors (deterministic, no LLM):**
| Extractor | Handles | Produces |
|---|---|---|
| `WholeFileExtractor` | anything | one `whole_file` span per editable file (exact today's behavior — the compat fallback) |
| `YamlJsonExtractor` | `*.yaml/*.yml/*.json` | a span per meaningful key: tool descriptions, `middlewares[]`, and **numeric/enum config → `HYPERPARAMS`** (with type+range in `meta`) |
| `MarkdownExtractor` | `*.md` | a span per `##` section; detects fenced/`Example:` blocks → **`DEMONSTRATIONS`**; rest → `INSTRUCTIONS`/`SKILLS`/`MEMORY` by filename |
| `PythonAstExtractor` | `*.py` | a span per function (`TOOL_CODE`), its docstring (`TOOL_DESCRIPTIONS`), module constants (`HYPERPARAMS`), and prompt-like string literals (`INSTRUCTIONS`); the agent loop if present (`CONTROL_FLOW`) |

**Cheap LLM tagger** runs *only* on spans the structural pass left ambiguous (e.g. "is this string a prompt or a label?") — one bounded JSON call over a span list, not blind file labeling. Reuses the existing `prompt_json` + a small schema.

**Framework adapters** (`studio/extract/adapters/`) — detected by signature files, applied first, override the generic pass:
- `nexau`: `code_agent.yaml` → tools[].description, middlewares[], and `max_iterations`/`max_tokens` as HYPERPARAMS (NOT the frozen model id); `systemprompt.md`, `tools/`, `middleware/`, `LongTermMEMORY.md`. **Replaces the hand-written `nexau_part_map()`** and adds the hyperparam/demo surfaces it lacked.
- `claude_code`: `CLAUDE.md`→INSTRUCTIONS; `.claude/commands/*`→SKILLS; `.claude/agents/*`→SUBAGENTS; `.claude/settings.json` hooks/permissions→CONTROL_FLOW + HYPERPARAMS; MCP config→TOOL_DESCRIPTIONS.
- `langgraph` / `openai_agents`: prompt templates→INSTRUCTIONS, tool fns→TOOL_*, graph nodes/edges & handoffs→CONTROL_FLOW/SUBAGENTS, model settings→HYPERPARAMS.
- `generic`: just the structural extractors + tagger (for unknown harnesses).

## 4. New high-leverage surfaces (the accuracy adds)

- **HYPERPARAMS** — *highest ROI / lowest effort.* Spans like `max_iterations`, retry counts, timeouts, temperature (model id stays frozen). `meta` carries `{type, min, max, choices}`; the strategist proposes a value, the shell validates it's in range. A one-token edit that can flip many tasks (e.g. raise the turn budget).
- **DEMONSTRATIONS** — a span holding few-shot examples (in a prompt section or a `demos.*` file). The strategist may add/edit/delete examples; new `demos` budget. DSPy-class lever.
- **CONTROL_FLOW** — the agent loop / stop / routing, where it lives *in the harness* (custom harnesses; many frameworks own it → marked non-editable there).
- **RETRIEVAL/CONTEXT** — spans that decide what enters context (a context-builder fn, a memory-injection block).

## 5. How each existing component changes (migration)

| Component | Today | Change | Compat |
|---|---|---|---|
| `parts.py` PartMap | type→files | keep; add `surface.py` alongside | unchanged |
| `mapper.map_harness` | LLM file→7types | becomes `extract.extract_surface` (structural+tagger+adapter) returning `SurfaceMap`; orchestrator calls `.to_partmap()` for legacy paths during migration | identical output for whole-file spans |
| `shell.enforce` | diff **files**, revert non-editable, budget per type by **file count** | diff **spans**: revert edits outside any editable span; budget per surface by **span count**; for whole_file spans this is byte-identical to today | whole-file path unchanged |
| `family_map` | families = combos of 7 types | unchanged — just sees more tags in the labels | unchanged |
| `strategist` | told do-not-touch files | told the **editable spans** (id, surface, locator, and for hyperparams the allowed range) + an "additive vs behavioral" hint | prompt-only change |
| `gate`/orchestrator | operate on whole harness | unchanged | unchanged |
| `content_hash` | all files | unchanged (file-level is fine for caching) | unchanged |

The only non-trivial code change is the **span-aware shell** (§ risks). Everything else is a new module + a prompt/threading change.

## 6. Phased rollout (each phase ships green tests)

- **P0 — scaffolding (no behavior change).** Add `surface.py` (`EditableSpan`, `SurfaceMap`, `WholeFileExtractor`, `to_partmap`/`from_partmap`). Orchestrator builds a `SurfaceMap` from the existing PartMap via `from_partmap`. Prove byte-identical behavior; all 100 tests pass.
- **P1 — structural spans + span-aware shell.** Add YAML/Markdown/AST extractors and the span-level shell diff. Gate it behind a flag; default still whole-file. New unit tests for span read/write round-trips and shell span-enforcement.
- **P2 — new surfaces (the accuracy win).** Add HYPERPARAMS + DEMONSTRATIONS with budgets + strategist hints + range validation. This is the first phase that should *move pass-rate*.
- **P3 — framework adapters.** `nexau` adapter first (replace `nexau_part_map`, add hyperparams/demos); then `claude_code`. Re-run the nexau head-to-head to measure the lift.
- **P4 — CONTROL_FLOW / RETRIEVAL** where the harness owns them.

## 7. Interaction with the gate/acceptance redesign (synergy)

The surface tag **is** the signal the acceptance rule needs. Tag each span edit:
- **Additive** (new TOOL_CODE/MIDDLEWARE/DEMONSTRATIONS, HYPERPARAMS-raise) → accept on **do-no-harm** (no regression on a held shard).
- **Behavioral** (INSTRUCTIONS rewrite, tool-behavior change, CONTROL_FLOW change) → require measured improvement beyond noise.

So the editable-surface work and the gate fix land together: surfaces make the right edits *reachable*; the acceptance rule lets them be *kept*.

## 8. Risks & mitigations

1. **Span-aware shell is the hard part** (mapping a coding-agent's free-form file edit back to "which spans changed"). Mitigate: diff at span boundaries via re-extraction (extract spans of original and candidate, compare per span); any changed byte outside a known editable span → revert that file region. Start with whole_file + YAML/MD (clean boundaries) before Python AST.
2. **Span-id drift across edits.** Anchor ids on names/keys, re-extract after each accept; never use line numbers as ids.
3. **Search-space blow-up** (too many tiny spans). Mitigate: per-surface span budget; the strategist still proposes one coherent edit; cap spans-per-file.
4. **Adapter rot** (frameworks change). Mitigate: adapters are thin and override-only; the generic structural path always works.
5. **Over-reach into frozen territory** (model id, secrets). The do-not-touch + `editable=False` on `llm_config.model` is enforced at the span level — stricter than today's file-level.

## 9. Testing

- Golden round-trip: extract → write a span → re-read equals expected, file structure preserved (YAML/MD/AST).
- Shell span-enforcement: an edit *outside* an editable span is reverted; *inside* survives; budgets counted per surface.
- Compat: `from_partmap(pm).to_partmap() == pm`; the nexau whole-file path reproduces today's family labels exactly.
- Adapter tests: nexau adapter yields the expected spans incl. HYPERPARAMS(`max_iterations`) and the frozen-model span as non-editable.
- End-to-end (Mock backend): a HYPERPARAMS edit and a DEMONSTRATIONS edit flow through strategist→shell→gate.

## 10. Effort (rough)

- P0: ~150 lines + tests (0.5 day). P1: ~300 lines (span shell + 3 extractors) + tests (1–2 days). P2: ~150 lines (2 surfaces + validation) (0.5–1 day). P3: ~120 lines/adapter (0.5 day each). Total to a production-credible v2: ~1 week, incremental, each phase shippable.

## 11. The first move

**P0 + the HYPERPARAMS surface for nexau (a slice of P2/P3).** It's small, it's the highest ROI (a `max_iterations`/turn-budget edit is one token and can flip hard tasks), and it immediately tests the thesis that "the win was unreachable, not unfindable." Then the span-aware shell (P1) unlocks everything else.
