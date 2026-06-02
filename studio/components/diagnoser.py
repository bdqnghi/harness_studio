"""Diagnoser (PRD §5.2): turn raw failures into causes + blame.

A Tier-B call: the round's failing tasks in, clustered failure patterns out, each
naming a root cause and blaming an editable part (or "unclear"). Routing rides on
this same call — no separate router. The blame is a hint, not a decision; the
gate still has the final say, so a wrong blame just wastes one strategy.
"""

from __future__ import annotations

from .. import schemas
from ..backends.base import Backend
from ..parts import PartType
from .runner import Failure

TAG = "diagnoser"

_BLAME_OPTIONS = [p.value for p in PartType] + ["unclear"]


def diagnose(
    backend: Backend, failures: list[Failure], *, model: str | None = None
) -> list[dict]:
    if not failures:
        return []
    listing = "\n".join(f"- {f.task_id}: {f.description}" for f in failures)
    prompt = (
        "These benchmark tasks failed. Cluster them by failure mode, infer the "
        "root cause of each cluster, and blame the harness part most likely "
        "responsible.\n\n"
        f"Blame must be one of: {', '.join(_BLAME_OPTIONS)}.\n\n"
        f"Failing tasks:\n{listing}\n\n"
        "Return a JSON array of clusters; each cluster has: pattern_id, "
        "description, root_cause, failing_task_ids (subset of the ids above), "
        "blamed_part, confidence (0-1)."
    )
    return backend.prompt_json(prompt, schemas.DIAGNOSIS, tag=TAG, model=model)
