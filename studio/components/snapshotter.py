"""Snapshotter (PRD §5.9): cheap rewind points.

The harness is text, so we save a full copy every round tagged with the round
number and the score at that point. The deep auditor rewinds to the last good
snapshot when a change turns out to be secretly worse (PRD §5.11).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..harness import Harness


@dataclass
class Snapshot:
    round_idx: int
    score: float
    path: Path


class Snapshotter:
    def __init__(self, snapshots_dir: Path) -> None:
        self.dir = Path(snapshots_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[Snapshot] = []

    def save(self, harness: Harness, round_idx: int, score: float) -> Snapshot:
        dest = self.dir / f"round_{round_idx:03d}"
        harness.copy_to(dest)
        (dest / ".snapshot.json").write_text(
            json.dumps({"round": round_idx, "score": score})
        )
        snap = Snapshot(round_idx, score, dest)
        self.snapshots.append(snap)
        return snap

    def best(self) -> Snapshot | None:
        return max(self.snapshots, key=lambda s: s.score) if self.snapshots else None

    def restore(self, snapshot: Snapshot, dest: Path) -> Harness:
        """Copy a snapshot back over ``dest`` (the live harness location)."""
        return Harness(snapshot.path).copy_to(dest)
