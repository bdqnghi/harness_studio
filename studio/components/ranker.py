"""Ranker (PRD §5.5): decide the testing order so the most promising goes first.

A Tier-B call. Critically a **pre-filter, not a decision**: a mis-ranked strategy
just gets tested first and rejected by the gate. Ranking affects efficiency
(fewer gate runs before a winner), never correctness.
"""

from __future__ import annotations

import json

from .. import schemas
from ..backends.base import Backend

TAG = "ranker"


def rank(
    backend: Backend, summaries: list[dict], *, model: str | None = None
) -> list[str]:
    """Return strategy_ids best-guess-first. Falls back to input order on any
    omission so every surviving strategy still gets its turn."""
    ids = [s["strategy_id"] for s in summaries]
    if len(ids) <= 1:
        return ids
    prompt = (
        "Order these strategies from most to least likely to improve the harness. "
        "This only sets testing order; the objective gate still decides.\n\n"
        f"Strategies:\n{json.dumps(summaries, indent=2)}\n\n"
        'Return JSON {"order": [strategy_ids, best first]}.'
    )
    data = backend.prompt_json(prompt, schemas.RANKING, tag=TAG, model=model)
    order = [i for i in data["order"] if i in ids]
    order += [i for i in ids if i not in order]  # append any the ranker dropped
    return order
