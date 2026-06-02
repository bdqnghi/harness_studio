"""Reviewer (PRD §5.4): prune obviously bad strategies before any testing.

A Tier-B call over *whole strategies* (not fragments): drop incoherent,
implausible, or known-dead (on the family map's do-not-repeat list) strategies.
It does not rank. Dropping a borderline-good strategy only costs a missed try —
the gate, not the reviewer, is the referee — so the reviewer errs toward keeping.
"""

from __future__ import annotations

import json

from .. import schemas
from ..backends.base import Backend

TAG = "reviewer"


def review(
    backend: Backend,
    summaries: list[dict],
    do_not_repeat: list[str],
    *,
    model: str | None = None,
) -> dict:
    """``summaries`` are dicts: {strategy_id, family_label, changed_parts, intent}.
    Returns {"keep": [ids], "drop": [{strategy_id, reason}]}."""
    if not summaries:
        return {"keep": [], "drop": []}
    dnr = "\n".join(f"- {f}" for f in do_not_repeat) or "(none)"
    prompt = (
        "Review these proposed strategies for fixing benchmark failures. Drop any "
        "that are incoherent, implausible, or belong to a falsified family on the "
        "do-not-repeat list. Keep everything else — the objective gate decides the "
        "rest, so do not over-prune.\n\n"
        f"Do-not-repeat families:\n{dnr}\n\n"
        f"Strategies:\n{json.dumps(summaries, indent=2)}\n\n"
        'Return JSON {"keep": [strategy_ids], "drop": [{"strategy_id","reason"}]}.'
    )
    return backend.prompt_json(prompt, schemas.REVIEW, tag=TAG, model=model)
