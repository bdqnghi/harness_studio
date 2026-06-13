"""Ideator (tree optimizer): cheap Tier-B hypothesis generation.

Two calls replace the classic path's n full Tier-A proposer runs:

* ``assign_directions`` — one router call per round. Failure signatures are
  free text and will not exact-match across rounds, so an LLM matches this
  round's patterns onto the existing direction nodes (or proposes new ones).
  Malformed output degrades to "every pattern is a new direction" — wasteful
  but never wrong.
* ``ideate`` — k text hypotheses (Mechanism/Hypothesis/Observable/Conflicts)
  under one chosen direction, conditioned on validated insights, the falsified
  ledger, and the pending frontier. Only the selected hypothesis is ever paid
  for with a Tier-A implementation run.
"""

from __future__ import annotations

from studio import schemas
from studio.backends.base import Backend
from studio.stages.optimize.idea_tree import Node

ROUTER_TAG = "direction-router"
IDEATE_TAG = "ideator"

# The exact header the falsified ledger is rendered under; tests grep for it.
CONSTRAINT_HEADER = "DO NOT RE-PROPOSE any of these falsified hypotheses:"


def assign_directions(
    backend: Backend, directions: list[Node], patterns: list[dict],
    *, model: str | None = None,
) -> list[dict]:
    """Map failure patterns onto direction nodes. Returns a list of
    ``{pattern_id, direction_id, new_title?, new_mechanism?}`` assignments
    (``direction_id == ""`` means: create a new direction)."""
    if not patterns:
        return []
    if directions:
        existing = "\n".join(
            f"- {d.id}: {d.title} (mechanism: {d.mechanism}; status: {d.status})"
            for d in directions
        )
    else:
        existing = "(none yet)"
    listing = "\n".join(
        f"- {p.get('pattern_id', '?')}: verifier_cause={p.get('verifier_cause', '')!r}, "
        f"agent_mechanism={p.get('agent_mechanism', '')!r}, "
        f"blamed_part={p.get('blamed_part', '')}, root_cause={p.get('root_cause', '')!r}"
        for p in patterns
    )
    prompt = (
        "You maintain a tree of improvement directions for a coding-agent "
        "harness. Assign each failure pattern below to the existing direction "
        "that addresses the same underlying failure mechanism, or propose a new "
        "direction when none fits.\n\n"
        f"Existing directions:\n{existing}\n\n"
        f"This round's failure patterns:\n{listing}\n\n"
        "Return assignments: one entry per pattern_id. Use the direction's id "
        'to reuse it, or direction_id "" with new_title (short, mechanism-level) '
        "and new_mechanism (one sentence) to create one. Group patterns sharing "
        "a mechanism into the same direction."
    )
    try:
        out = backend.prompt_json(prompt, schemas.DIRECTION_ASSIGN, tag=ROUTER_TAG, model=model)
        assignments = out.get("assignments", [])
    except Exception:  # noqa: BLE001 — router failure must not kill the round
        assignments = []
    routed = {a.get("pattern_id") for a in assignments if isinstance(a, dict)}
    # Deterministic fallback: any unrouted pattern becomes a new direction.
    for p in patterns:
        pid = p.get("pattern_id", "?")
        if pid not in routed:
            assignments.append({
                "pattern_id": pid, "direction_id": "",
                "new_title": (p.get("root_cause") or pid)[:80],
                "new_mechanism": p.get("agent_mechanism", ""),
            })
    return assignments


def ideate(
    backend: Backend, direction: Node, *, diagnosis: list[dict],
    validated_insights: list[str], falsified: list[str], pending: list[str],
    k: int = 4, model: str | None = None, trace_evidence: dict | None = None,
) -> list[dict]:
    """k hypotheses under ``direction``: title/mechanism/hypothesis/observable.

    ``trace_evidence`` (optional {task_id: excerpt}) grounds ideation in the
    actual failure transcripts, not just the diagnosis summary."""
    evidence = "\n".join(
        f"- [{d.get('pattern_id', '?')}] {d.get('root_cause', '')} "
        f"(verifier: {d.get('verifier_cause', '')}; agent: {d.get('agent_mechanism', '')}; "
        f"tasks: {', '.join(d.get('failing_task_ids', [])[:5])})"
        for d in diagnosis
    ) or "(no fresh failure evidence this round)"
    insights = "\n".join(f"- {i}" for i in validated_insights) or "(none yet)"
    sections = [
        f"Direction under exploration: {direction.title}\n"
        f"Mechanism: {direction.mechanism}",
        f"This round's failure evidence:\n{evidence}",
        f"Validated insights from already-tested ideas (build on these):\n{insights}",
    ]
    if trace_evidence:
        items = [(t, v) for t, v in trace_evidence.items() if v][:3]
        if items:
            budget = max(300, 2000 // len(items))
            sections.append(
                "Concrete failing-task transcripts (ground your ideas in these):\n"
                + "\n".join(f"[task {t}] {str(v).strip()[:budget]}" for t, v in items)
            )
    if falsified:
        sections.append(
            CONSTRAINT_HEADER + "\n" + "\n".join(f"- {c}" for c in falsified)
        )
    if pending:
        sections.append(
            "Already queued — avoid duplicates:\n" + "\n".join(f"- {t}" for t in pending)
        )
    sections.append(
        f"Propose exactly {k} DISTINCT hypotheses for improving the harness "
        "within this direction. Each must be a real mechanism, not a parameter "
        "tweak or a 'try harder' instruction, and must differ from the others "
        "by mechanism class. For each give: title (short), mechanism (how the "
        "harness change works), hypothesis (the concrete edit to make), "
        "observable (the specific measurable effect on the failing tasks if "
        "the hypothesis is right), conflicts (optional: what it might break)."
    )
    out = backend.prompt_json(
        "\n\n".join(sections), schemas.HYPOTHESES, tag=IDEATE_TAG, model=model,
    )
    return list(out.get("hypotheses", []))[:k]
