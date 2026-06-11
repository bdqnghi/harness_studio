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
    """Sizes for the four task piles (PRD §6). Practice is sampled per round."""

    practice: int = 12
    judging: int = 16
    audit: int = 24
    final_exam: int = 24


@dataclass
class GateConfig:
    """Gate tuning (PRD §5.8)."""

    borderline_extra_runs: int = 5  # capped re-runs for in-band decisions


@dataclass
class EditConfig:
    """Edit-discipline tuning (PRD §5.6, §5.7)."""

    budget_per_part: int = 3  # max changed files per part type in one strategy
    allow_repair: bool = True  # one structural-check repair attempt (PRD §11 Q6)


@dataclass
class LoopConfig:
    """Inner/outer loop tuning."""

    rounds: int = 8
    segment_length: int = 10  # rounds per segment (outer loop); bounded by meta
    wobble_runs: int = 5  # repeated runs at setup to measure the noise floor
    strategies_per_round: int = 3  # competing strategies the Strategist proposes


@dataclass
class HealthConfig:
    """Thresholds for the health monitor (PRD §7)."""

    empty_round_limit: int = 3  # consecutive empty rounds before flagging
    gate_rejection_limit: int = 5  # consecutive gate rejections before flagging


@dataclass
class EvalPlanConfig:
    """Power-based, calibration-aware split tuning (splitter.choose_eval_plan)."""

    adaptive: bool = False        # use choose_eval_plan instead of fixed piles
    z: float = 1.96               # confidence for power sizing
    delta_step: float = 0.12      # per-round effect the gate must resolve (coarse)
    delta_final: float = 0.05     # effect the test/CV verdict must resolve
    val_floor: int = 8
    val_budget_cap: int = 16      # gate-size ceiling -> n_val ~constant across N
    heavy_sec: float = 3600.0     # tasks at/above this stay out of the every-round gate
    n_folds: int = 5
    calibration_path: str = ""    # reuse a cached Calibration if set


@dataclass
class Config:
    seed: int = 0
    noise_per_mille: int = 0  # injected toy wobble; 0 for exact tests
    cache: bool = True  # cache benchmark scores within a segment (PRD §8)
    piles: PileConfig = field(default_factory=PileConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    edits: EditConfig = field(default_factory=EditConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    eval_plan: EvalPlanConfig = field(default_factory=EvalPlanConfig)

    # --- (de)serialization ---

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        return cls(
            seed=data.get("seed", 0),
            noise_per_mille=data.get("noise_per_mille", 0),
            cache=data.get("cache", True),
            piles=PileConfig(**data.get("piles", {})),
            gate=GateConfig(**data.get("gate", {})),
            loop=LoopConfig(**data.get("loop", {})),
            edits=EditConfig(**data.get("edits", {})),
            health=HealthConfig(**data.get("health", {})),
            eval_plan=EvalPlanConfig(**data.get("eval_plan", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            import yaml  # optional dependency

            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))
