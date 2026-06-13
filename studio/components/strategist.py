"""Strategist (PRD §5.3) — the core proposer, a Tier-A coding agent.

Each round it produces several *competing whole-strategies*. A strategy is one
complete, internally-coordinated proposal (it may touch several parts at once) —
the unit that competes at the gate. We realize each as its own candidate copy of
the harness, edited in place by one coding-agent run, so every strategy is a
coordinated fix by a single mind (PRD §5.3's "one agent, not N per-part fixers").

Strategies are diversified by *angle* (which the family map and diagnosis
suggest), not decomposed by part. The family label is derived later from the
parts a strategy actually changed (in the orchestrator), so we never depend on
the agent reporting structured metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..backends.base import AgentResult, Backend
from ..harness import Harness

TAG = "strategist"
BUILD_TAG = "builder"
SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "strategist" / "SKILL.md"

# Guidance for the GENERATE-from-scratch mode (workspace starts empty). Same
# engine as editing — only the situation differs: there is nothing to edit yet,
# so the agent writes a complete, runnable harness rather than a minimal diff.
_BUILD_SKILL = """You are a coding agent building a NEW agent harness FROM SCRATCH. The workspace is empty.

- Create a minimal but COMPLETE and runnable harness for the task: the file(s) the runtime
  will execute (exactly as the runner contract requires), the tool wiring, and a working
  control loop / operating instructions. It must boot and run end-to-end — no TODOs, no stubs.
- Make sensible engineering choices; prefer the simplest design that fully satisfies the
  contract. Use read_file/list_dir to inspect anything you write.
- When the harness is complete and runnable, call complete_task with a short summary.
"""


def load_skill() -> str:
    return SKILL_PATH.read_text()


def _build_instruction(brief) -> str:
    tools = "\n".join(f"- {t.name}{_paren(t.signature)}: {t.doc}" for t in (brief.tools or [])) \
        or "(the runtime provides the tools; wire them as the contract describes)"
    notes = f"\nNotes: {brief.extra_notes}\n" if getattr(brief, "extra_notes", "") else ""
    return (
        "Build a new, runnable agent harness FROM SCRATCH for the task below. The workspace "
        "is empty — create the files the runtime executes, wire the tools, and a working "
        "control loop / operating policy. It MUST boot and run end-to-end.\n\n"
        f"Task domain: {brief.domain}\n"
        f"IO contract: {brief.io_contract}\n"
        f"Available tools:\n{tools}\n"
        f"Runner contract (what the benchmark will execute — your harness MUST expose this):\n"
        f"{brief.runner_contract or '(produce the obvious runnable entrypoint for this task)'}\n"
        f"{notes}\n"
        "Create the files now, then call complete_task."
    )


def _paren(sig: str) -> str:
    """Render just the arg list of a signature for a prompt listing."""
    if "(" in sig:
        return "(" + sig.split("(", 1)[1].split(")")[0] + ")"
    return ""


def build_harness(
    backend: Backend,
    workspace: Path,
    brief,
    *,
    validate=None,
    max_attempts: int = 2,
    do_not_touch: list[str] | None = None,
    model: str | None = None,
) -> Harness:
    """Generate a round-0 harness by running the SAME coding agent on an empty
    workspace (no templates). If ``validate(harness) -> (ok, err)`` is given
    (e.g. the benchmark's boot_check), retry up to ``max_attempts`` times,
    feeding the error back — the agent decides how to fix its own output, exactly
    as in the edit loop."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    harness = Harness(workspace)
    for rel, content in (getattr(brief, "seed_files", None) or {}).items():
        harness.write_file(rel, content)

    instruction = _build_instruction(brief)
    for attempt in range(max(1, max_attempts)):
        backend.run_agent(instruction, workspace=workspace, skill=_BUILD_SKILL,
                          tag=BUILD_TAG, model=model)
        if validate is None:
            return harness
        ok, err = validate(harness)
        if ok:
            return harness
        instruction = (
            "The harness you generated does not boot/run yet. Fix it so it boots and runs "
            f"end-to-end, keeping the runner contract.\n\nError:\n{err}\n"
        )
    return harness  # caller still boot_checks and surfaces a clear error if invalid


@dataclass
class Strategy:
    """One competing proposal: a candidate harness plus what it attempted."""

    strategy_id: str
    candidate: Harness
    intent: str
    family_label: str = ""  # set by the orchestrator from the parts it changed
    changed_parts: dict = field(default_factory=dict)
    result: AgentResult | None = None


def _format_diagnosis(diagnosis: list[dict]) -> str:
    if not diagnosis:
        return "(no specific diagnosis available)"
    lines = []
    for d in diagnosis:
        lines.append(
            f"- [{d.get('pattern_id', '?')}] {d.get('root_cause', d.get('description', ''))} "
            f"(blame: {d.get('blamed_part', 'unclear')}; tasks: {', '.join(d.get('failing_task_ids', [])[:5])})"
        )
    return "\n".join(lines)


def _editable_block(editable_files: list[str] | None) -> str:
    """Tell the agent EXACTLY which paths it may change — critical when the
    editable surface is small (e.g. a single prose policy). Anything written
    outside this whitelist is reverted by the shell, so an edit that lands
    elsewhere is silently dropped. A path ending in ``/`` is a directory the
    agent may add files under; otherwise it must edit that existing file in
    place (do NOT create new files)."""
    if not editable_files:
        return ""
    files = ", ".join(editable_files)
    dirs = [f for f in editable_files if f.endswith("/")]
    rule = (
        " You may add files only under the directory entries; for plain files, "
        "edit them in place."
        if dirs else
        " These are PLAIN FILES — edit them IN PLACE. Do NOT create new files: "
        "anything outside this list is discarded, so your change must live in "
        "these files (e.g. for a prose policy, add/rewrite the relevant rules)."
    )
    return (f"\n\nEDITABLE FILES (you may ONLY change these): {files}.{rule}")


def _format_evidence(evidence: dict | None, *, char_budget: int = 4000) -> str:
    """Render the failing-task evidence (verifier output + transcript windows)
    for the editor — the "read the failure before you fix it" payload. Empty
    string when no evidence, so prompts stay byte-identical to before."""
    if not evidence:
        return ""
    items = [(t, v) for t, v in evidence.items() if v]
    if not items:
        return ""
    per = max(400, char_budget // len(items))
    lines = ["\n\nFailure evidence (verifier output + transcript windows) — "
             "READ THIS before editing; fix what it actually shows:"]
    for tid, text in items:
        lines.append(f"\n[task {tid}]\n{str(text).strip()[:per]}")
    return "\n".join(lines)


def _format_localization(localization: list[dict] | None) -> str:
    """Render evidence-grounded edit targets: which file/span to change and the
    cited evidence. Empty string when none (prompt unchanged)."""
    if not localization:
        return ""
    lines = ["\n\nLocalized edit target(s) — evidence-grounded; make the change here:"]
    for t in localization:
        loc, kind = t.get("target_locator", ""), t.get("change_kind", "")
        head = f"\n- file: {t.get('target_file', '')}"
        if loc:
            head += f" @ {loc}"
        if kind:
            head += f" [{kind}]"
        lines.append(head)
        if t.get("current_text"):
            body = str(t["current_text"]).strip()[:600].replace("\n", "\n    ")
            lines.append(f"  current text to change:\n    {body}")
        if t.get("rationale"):
            lines.append(f"  why: {str(t['rationale'])[:300]}")
        for e in (t.get("evidence") or [])[:3]:
            lines.append(f"  evidence [{e.get('task_id', '')}]: {str(e.get('quote', ''))[:200]}")
    return "\n".join(lines)



def implement_hypothesis(
    backend: Backend,
    base: Harness,
    cand_dir: Path,
    node,
    diagnosis: list[dict],
    *,
    strategy_id: str,
    do_not_touch: list[str] | None = None,
    validated_insights: list[str] | None = None,
    model: str | None = None,
    editable_files: list[str] | None = None,
    localization: list[dict] | None = None,
    evidence: dict | None = None,
    evidence_dir: Path | None = None,
) -> Strategy:
    """Stage 2 of the tree path: one Tier-A run implementing ONE pre-selected
    text hypothesis (an idea_tree Node). The hypothesis is the contract — the
    agent implements it, it does not get to substitute its own idea; competing
    ideas were already compared cheaply as text at stage 1."""
    dnt = ", ".join(do_not_touch or []) or "(none)"
    insights = "\n".join(f"- {i}" for i in (validated_insights or []))
    insights_block = f"\n\nValidated insights from sibling ideas:\n{insights}" if insights else ""
    instruction = (
        "Implement EXACTLY this hypothesis — it is fixed and non-negotiable. "
        "Make the smallest coherent change that realizes it and keep the "
        "harness booting. Realize the hypothesis WITHIN the editable files "
        "below — if the only editable file is a prose policy, express the idea "
        "as added/rewritten policy rules, NOT as new code.\n\n"
        f"Hypothesis: {node.title}\n"
        f"Mechanism: {node.mechanism}\n"
        f"Edit to make: {node.hypothesis}\n"
        f"Predicted observable effect: {node.observable}\n\n"
        f"Diagnosis of this round's failures:\n{_format_diagnosis(diagnosis)}\n\n"
        f"Do-not-touch files: {dnt}{_editable_block(editable_files)}"
        f"{_format_localization(localization)}{_format_evidence(evidence)}{insights_block}"
    )
    candidate = base.copy_to(Path(cand_dir))
    result = backend.run_agent(
        instruction, workspace=Path(cand_dir), skill=load_skill(), tag=TAG, model=model,
        read_dirs=[Path(evidence_dir)] if evidence_dir else None,
    )
    return Strategy(strategy_id=strategy_id, candidate=candidate,
                    intent=node.title, result=result)


def repair(
    backend: Backend,
    candidate_dir: Path,
    error: str,
    *,
    do_not_touch: list[str] | None = None,
    model: str | None = None,
) -> AgentResult:
    """One bounded fix attempt: the harness didn't boot; here is the error."""
    dnt = ", ".join(do_not_touch or []) or "(none)"
    instruction = (
        "Your previous edit left the harness unable to compile/boot. Fix it with "
        "the smallest possible change so it boots again, preserving the intent of "
        f"the edit.\n\nDo-not-touch files: {dnt}\n\nError:\n{error}\n"
    )
    return backend.run_agent(
        instruction, workspace=candidate_dir, skill=load_skill(),
        tag="strategist-repair", model=model,
    )


def family_label(changed_parts: dict) -> str:
    """Derive a strategy family from the set of part types it changed.

    Families are classes of approach (PRD §5.10.2): grain coarse enough that a
    lesson generalizes ("instructions+tool_code fixes"), not edit-specific."""
    if not changed_parts:
        return "none"
    return "+".join(sorted(p.value for p in changed_parts))
