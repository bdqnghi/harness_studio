"""Diagnoser (PRD §5.2): turn raw failures into causes + blame.

A Tier-B call: the round's failing tasks in, clustered failure patterns out, each
naming a root cause and blaming an editable part (or "unclear"). Routing rides on
this same call — no separate router. The blame is a hint, not a decision; the
acceptance still has the final say, so a wrong blame just wastes one strategy.
"""

from __future__ import annotations

from studio import schemas
from studio.backends.base import Backend
from studio.core.parts import PartType
from studio.stages.optimize.runner import Failure

TAG = "diagnoser"

_BLAME_OPTIONS = [p.value for p in PartType] + ["unclear"]


def diagnose(
    backend: Backend, failures: list[Failure], *, model: str | None = None
) -> list[dict]:
    if not failures:
        return []

    def _entry(f: Failure) -> str:
        head = f"- {f.task_id}: {f.description}"
        if f.trace:
            # Indent the trace so the cluster listing stays readable.
            body = "\n".join("    " + ln for ln in f.trace.splitlines())
            return f"{head}\n  failure evidence:\n{body}"
        return head

    listing = "\n".join(_entry(f) for f in failures)
    prompt = (
        "These benchmark tasks failed. Use the failure evidence (verifier output "
        "and the agent's last actions) to cluster them by failure mode, infer the "
        "root cause of each cluster, and blame the harness part most likely "
        "responsible.\n\n"
        f"Blame must be one of: {', '.join(_BLAME_OPTIONS)}.\n\n"
        f"Failing tasks:\n{listing}\n\n"
        "Return a JSON array of clusters; each cluster has: pattern_id, "
        "description, root_cause, failing_task_ids (subset of the ids above), "
        "blamed_part, confidence (0-1), and a failure signature: verifier_cause "
        "(what the verifier mechanically observed), agent_mechanism (what the "
        "agent did or failed to do that produced it), addressable (boolean; "
        "false when no harness edit could plausibly fix it — task-specific "
        "difficulty, infrastructure flake, or raw model-capability limits)."
    )
    out = backend.prompt_json(prompt, schemas.DIAGNOSIS, tag=TAG, model=model)
    for d in out:
        if isinstance(d, dict):  # default-fill so downstream code can rely on the keys
            d.setdefault("verifier_cause", "")
            d.setdefault("agent_mechanism", "")
            d.setdefault("addressable", True)
    return out
