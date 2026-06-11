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
    """Power-based, calibration-aware split tuning (splitter.choose_split)."""

    adaptive: bool = False        # use choose_split instead of fixed piles
    round_size: int = 32          # tasks run per round (the SGD mini-batch)
    z: float = 1.96               # confidence for power sizing
    delta_round: float = 0.12     # per-round effect the gate must resolve (coarse)
    val_floor: int = 8            # min stable gate (judging) size
    reg_floor: int = 16           # min regression (do-no-harm) size
    reg_cap: int = 32             # max regression size -> ~constant across N
    pool_mult: int = 4            # held-in pool = pool_mult * round_size
    pool_cap: int = 256           # held-in pool ceiling -> ~constant across N
    test_floor: int = 25          # min locked-test size for a trustworthy verdict
    test_budget_cap: int = 0      # >0: grade only a representative subsample of test
    heavy_sec: float = 3600.0     # tasks at/above this go ONLY to the locked test
    calibration_k: int = 3        # repeated held-in baseline rollouts for noise
    opt_k: int = 1                # rollouts per gate check (cheap, per-round)
    test_k: int = 3               # rollouts for the locked-test verdict (trustworthy)
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
        eval_plan = dict(data.get("eval_plan", {}))
        eval_plan.pop("delta_final", None)  # removed: actual detectable_final is reported
        return cls(
            seed=data.get("seed", 0),
            noise_per_mille=data.get("noise_per_mille", 0),
            cache=data.get("cache", True),
            piles=PileConfig(**data.get("piles", {})),
            gate=GateConfig(**data.get("gate", {})),
            loop=LoopConfig(**data.get("loop", {})),
            edits=EditConfig(**data.get("edits", {})),
            health=HealthConfig(**data.get("health", {})),
            eval_plan=EvalPlanConfig(**eval_plan),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            import yaml  # optional dependency

            return cls.from_dict(yaml.safe_load(text))
        return cls.from_dict(json.loads(text))
