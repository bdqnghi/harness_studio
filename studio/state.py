"""Run state: the directory layout and the durable records the loop produces.

The Orchestrator owns one WorkspaceState. It holds the live harness location, the
scratch areas (candidates, snapshots), and the append-only **evidence record** —
the per-round outcomes the deep auditor and meta-agent later read (PRD §5.10,
§5.11). Memory is compressed, not replayed (PRD principle 6): we keep the current
harness + the evidence record, never a full transcript.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RoundOutcome:
    """One inner-loop round's result — a row in the evidence record."""

    round_idx: int
    accepted: bool
    gain: float
    old_score: float
    new_score: float
    family_label: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "round": self.round_idx,
            "accepted": self.accepted,
            "gain": round(self.gain, 4),
            "old_score": round(self.old_score, 4),
            "new_score": round(self.new_score, 4),
            "family_label": self.family_label,
            "note": self.note,
        }


@dataclass
class HealthCounters:
    """Signals the orchestrator watches (PRD §7)."""

    empty_rounds: int = 0  # consecutive rounds where everything was dropped
    gate_rejections: int = 0  # consecutive gate rejections
    reward_hack_incidents: int = 0


@dataclass
class WorkspaceState:
    """Filesystem layout + in-memory records for one optimization run."""

    root: Path
    wobble: float = 0.0
    score_history: list[float] = field(default_factory=list)
    evidence: list[RoundOutcome] = field(default_factory=list)
    health: HealthCounters = field(default_factory=HealthCounters)
    health_log: list[str] = field(default_factory=list)  # health signals raised
    avoid_list: list[str] = field(default_factory=list)  # known-dead edits/errors

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        for d in (self.harness_dir, self.candidates_dir, self.snapshots_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- standard locations ---

    @property
    def harness_dir(self) -> Path:
        return self.root / "harness"

    @property
    def candidates_dir(self) -> Path:
        return self.root / "candidates"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def family_map_path(self) -> Path:
        return self.root / "family_map.md"

    @property
    def evidence_path(self) -> Path:
        return self.root / "evidence.jsonl"

    @property
    def progress_path(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def health_log_path(self) -> Path:
        return self.root / "health.log"

    # --- recording ---

    def record(self, outcome: RoundOutcome) -> None:
        self.evidence.append(outcome)
        with self.evidence_path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), **outcome.to_dict()}) + "\n")

    def log_health(self, line: str) -> None:
        """Health signals must survive the process (a halted run's last words)."""
        self.health_log.append(line)
        try:
            with self.health_log_path.open("a") as f:
                f.write(f"{time.time():.0f} {line}\n")
        except OSError:
            pass
