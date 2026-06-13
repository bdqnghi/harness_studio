"""Typed configuration for a run.

Plain dataclasses with sensible defaults; load from JSON (and YAML if PyYAML is
installed) so a run is reproducible from a single file. Fields grow per milestone;
M0 uses the loop/gate/pile fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PileConfig:
    """Sizes for the task sets. ``round_size`` is the per-round batch sampled
    from held_in; ``regression``/``held_out`` size the fixed-fallback split."""

    round_size: int = 12      # per-round held-in batch (the SGD mini-batch)
    regression: int = 0       # disjoint do-no-harm set (0 = none in the fixed fallback)
    held_out: int = 24        # locked, graded once


@dataclass
class GateConfig:
    """Gate tuning (PRD §5.8)."""

    borderline_extra_runs: int = 5  # capped re-runs for in-band decisions
    strict_dual: bool = False  # require EACH slice not-regress (default: net pooled gain)


@dataclass
class EditConfig:
    """Edit-discipline tuning (PRD §5.6, §5.7)."""

    budget_per_part: int = 3  # max changed files per part type in one strategy
    allow_repair: bool = True  # one structural-check repair attempt (PRD §11 Q6)


@dataclass
class LoopConfig:
    """Inner/outer loop tuning."""

    rounds: int = 8
    segment_length: int = 10  # rounds per segment (the deep-audit/rewind boundary)
    wobble_runs: int = 5  # repeated runs at setup to measure the noise floor
    hypotheses_per_direction: int = 4  # text hypotheses per ideation call
    # Context localization (stages/optimize/localizer.py): "off" (diagnosis-only, the
    # legacy behavior) | "inline" | "agentic" | "auto" (pick by difficulty).
    localizer: str = "off"


@dataclass
class HealthConfig:
    """Thresholds for the health monitor (PRD §7)."""

    empty_round_limit: int = 3  # consecutive empty rounds before flagging
    gate_rejection_limit: int = 5  # consecutive gate rejections before flagging


@dataclass
class Config:
    seed: int = 0
    noise_per_mille: int = 0  # injected toy wobble; 0 for exact tests
    cache: bool = True  # cache benchmark scores within a segment (PRD §8)
    score_cache: str = ""  # disk-backed score cache (JSONL); "" = memory only
    piles: PileConfig = field(default_factory=PileConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    edits: EditConfig = field(default_factory=EditConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    # --- (de)serialization ---

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(
            seed=data.get("seed", 0),
            noise_per_mille=data.get("noise_per_mille", 0),
            cache=data.get("cache", True),
            score_cache=data.get("score_cache", ""),
            piles=PileConfig(**data.get("piles", {})),
            gate=GateConfig(**data.get("gate", {})),
            loop=LoopConfig(**data.get("loop", {})),
            edits=EditConfig(**data.get("edits", {})),
            health=HealthConfig(**data.get("health", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            import yaml  # optional dependency

            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))
