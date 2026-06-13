"""Continuous run observability: an append-only event stream.

A multi-hour optimization run is otherwise silent until it finishes; the
ProgressLog gives every round a tail-able trail::

    tail -f <workspace>/progress.jsonl

One JSON object per line, always with ``ts`` (unix seconds) and ``event``.
Emission is best-effort and never raises into the loop: observability must not
be able to kill a run.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path


class ProgressLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def emit(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass


def decision_dict(decision) -> dict:
    """A GateDecision as a JSON-safe dict with floats rounded for readability."""
    out = dataclasses.asdict(decision)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}
