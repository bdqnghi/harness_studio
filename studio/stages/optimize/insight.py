"""Insight distillation (tree optimizer): the lesson, not the log.

Arbor's ablations put insight propagation above the tree itself (81.8% with
both vs 54.5% without insights): future ideas inherit *why* past ones won or
lost instead of rediscovering it. After each gate decision we distill one
<=200-word lesson onto the tested node; when a direction gains a definitive
result, a second short call refreshes the direction's summary.

Both calls are best-effort Tier-B: a distillation failure degrades to an
empty insight, never a failed round.
"""

from __future__ import annotations

from studio import schemas
from studio.backends.base import Backend
from studio.stages.optimize.idea_tree import Node

TAG = "insight"
DIRECTION_TAG = "insight-direction"


def distill(
    backend: Backend, node: Node, decision, diagnosis: list[dict],
    *, model: str | None = None,
) -> str:
    """The <=200-word lesson from testing one hypothesis at the gate."""
    verdict = "ACCEPTED" if decision.accept else "REJECTED"
    evidence = "; ".join(
        f"{d.get('pattern_id', '?')}: {d.get('root_cause', '')}" for d in diagnosis[:4]
    )
    prompt = (
        "A hypothesis for improving a coding-agent harness was implemented and "
        "measured. Distill the transferable lesson in under 200 words: WHY it "
        "won or lost (mechanism-level, not numbers), and what a sibling idea in "
        "the same direction should do differently or build on.\n\n"
        f"Hypothesis: {node.title} — {node.hypothesis}\n"
        f"Predicted observable: {node.observable}\n"
        f"Gate verdict: {verdict} (judging gain {decision.gain:+.3f}, "
        f"regression gain {decision.regression_gain:+.3f}, "
        f"runs used {decision.runs_used}; {decision.reason})\n"
        f"Failure evidence it targeted: {evidence}\n"
    )
    try:
        out = backend.prompt_json(prompt, schemas.INSIGHT, tag=TAG, model=model)
        return str(out.get("insight", ""))
    except Exception:  # noqa: BLE001 — observability of lessons must not kill a round
        return ""


def summarize_direction(
    backend: Backend, direction: Node, tested_children: list[Node],
    *, model: str | None = None,
) -> str:
    """Refresh the direction's <=200-word summary from its children's lessons.
    Called only when a child reaches a definitive status (accepted/falsified)."""
    listing = "\n".join(
        f"- {n.title} [{n.status}]: {n.insight or '(no insight)'}"
        for n in tested_children
    )
    prompt = (
        "Summarize what has been LEARNED about this improvement direction from "
        "its tested hypotheses, in under 200 words. State what works, what is "
        "falsified, and the most promising unexplored angle.\n\n"
        f"Direction: {direction.title} — {direction.mechanism}\n"
        f"Tested hypotheses:\n{listing}\n"
    )
    try:
        out = backend.prompt_json(prompt, schemas.INSIGHT, tag=DIRECTION_TAG, model=model)
        return str(out.get("insight", ""))
    except Exception:  # noqa: BLE001
        return ""
