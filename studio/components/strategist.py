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
SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "strategist" / "SKILL.md"


def load_skill() -> str:
    return SKILL_PATH.read_text()


@dataclass
class Strategy:
    """One competing proposal: a candidate harness plus what it attempted."""

    strategy_id: str
    candidate: Harness
    intent: str
    family_label: str = ""  # set by the orchestrator from the parts it changed
    changed_parts: dict = field(default_factory=dict)
    result: AgentResult | None = None


# --- diversification: distinct angles for the competing strategies ---

def diversification_hints(diagnosis: list[dict], n: int) -> list[str]:
    blamed = []
    for d in diagnosis:
        part = d.get("blamed_part")
        if part and part != "unclear" and part not in blamed:
            blamed.append(part)
    hints = []
    if blamed:
        hints.append(f"Fix the root cause directly by editing the blamed part(s): {', '.join(blamed)}.")
    hints.append("Address the failures by clarifying the instructions (add an explicit rule or example).")
    hints.append("Address the failures in the tool code or by adding a guard/middleware safeguard.")
    hints.append("Make a different, minimal coordinated change than the obvious one.")
    # de-dup, pad to n
    seen, out = set(), []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    while len(out) < n:
        out.append(out[len(out) % len(hints)])
    return out[:n]


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


def _instruction(diagnosis, hint, do_not_touch, family_map_text) -> str:
    dnt = ", ".join(do_not_touch or []) or "(none)"
    fm = f"\n\nStrategy-family map (prefer 'works', avoid 'falsified'):\n{family_map_text}" if family_map_text else ""
    return (
        "Improve this harness so it passes the failing tasks. Make ONE coherent, "
        "minimal strategy. You may touch several parts if they form one coordinated "
        f"fix, but keep it small and keep the harness booting.\n\n"
        f"Approach for this strategy: {hint}\n\n"
        f"Diagnosis of this round's failures:\n{_format_diagnosis(diagnosis)}\n\n"
        f"Do-not-touch files: {dnt}{fm}"
    )


def propose_many(
    backend: Backend,
    base: Harness,
    round_dir: Path,
    diagnosis: list[dict],
    *,
    n: int,
    id_prefix: str,
    do_not_touch: list[str] | None = None,
    family_map_text: str = "",
    model: str | None = None,
) -> list[Strategy]:
    """Run the coding agent ``n`` times (distinct angles) → ``n`` candidate strategies."""
    hints = diversification_hints(diagnosis, n)
    strategies: list[Strategy] = []
    for i, hint in enumerate(hints):
        cand_dir = Path(round_dir) / f"strategy_{i}"
        candidate = base.copy_to(cand_dir)
        result = backend.run_agent(
            _instruction(diagnosis, hint, do_not_touch, family_map_text),
            workspace=cand_dir, skill=load_skill(), tag=TAG, model=model,
        )
        strategies.append(
            Strategy(strategy_id=f"{id_prefix}s{i}", candidate=candidate,
                     intent=hint, result=result)
        )
    return strategies


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
