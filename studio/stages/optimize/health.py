"""Health monitor (PRD §7): watch the loop's health signals and name responses.

Each signal that crosses a threshold maps to a defined response. This component
only *detects and names*; the orchestrator acts (feed more context, trigger an
early meta intervention, or halt on a reward-hacking attempt). The reward-hack
signal is raised eagerly by the InstrumentedBenchmark; the streak signals are
assessed here from the running counters.
"""

from __future__ import annotations

from dataclasses import dataclass

from studio.config import HealthConfig
from studio.core.state import HealthCounters


@dataclass
class HealthSignal:
    name: str
    detail: str
    response: str


def assess(health: HealthCounters, cfg: HealthConfig) -> list[HealthSignal]:
    signals: list[HealthSignal] = []
    if health.empty_rounds >= cfg.empty_round_limit:
        signals.append(HealthSignal(
            "empty_rounds",
            f"{health.empty_rounds} consecutive empty rounds",
            "feed the Strategist more context; re-map or stop if it persists",
        ))
    if health.gate_rejections >= cfg.gate_rejection_limit:
        signals.append(HealthSignal(
            "gate_rejection_streak",
            f"{health.gate_rejections} consecutive gate rejections",
            "trigger an early meta-agent intervention (pivot directive)",
        ))
    if health.reward_hack_incidents > 0:
        signals.append(HealthSignal(
            "reward_hack",
            f"{health.reward_hack_incidents} reward-hacking incident(s)",
            "halt — the gate isolation should prevent this; flag and stop",
        ))
    return signals
