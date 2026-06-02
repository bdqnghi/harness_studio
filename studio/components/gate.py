"""The gate (PRD §5.8): the referee, and the only thing that changes the harness.

It compares the harness old-way vs new-way on the judging set and decides
keep/reject with a three-way, noise-aware rule:

  * gain  > wobble        -> clearly better -> accept
  * gain <= 0             -> clearly not    -> reject (regression if gain < 0)
  * 0 < gain <= wobble    -> borderline     -> re-run the *contested* tasks a few
                              more times (capped) to average out noise, then
                              accept iff the averaged gain is positive.

``gain`` is the mean over judging tasks of the paired difference
``score(new) - score(old)``.

Protection (PRD §3): the gate is constructed with a benchmark only — never a
Backend. No AI helper can reach the evaluator or write scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..benchmark.base import Benchmark
from ..harness import Harness


@dataclass
class GateDecision:
    accept: bool
    gain: float
    old_score: float
    new_score: float
    regressed: bool = False
    borderline: bool = False
    runs_used: int = 1
    reason: str = ""


def _mean(d: dict[str, float]) -> float:
    return sum(d.values()) / len(d) if d else 0.0


class Gate:
    def __init__(
        self,
        benchmark: Benchmark,
        judging_tasks: list[str],
        wobble: float,
        *,
        borderline_extra_runs: int = 5,
    ) -> None:
        self.benchmark = benchmark
        self.judging = list(judging_tasks)
        self.wobble = max(0.0, wobble)
        self.extra = max(0, borderline_extra_runs)

    def evaluate(self, old: Harness, new: Harness) -> GateDecision:
        old_s = self.benchmark.run(old, self.judging, run_idx=0)
        new_s = self.benchmark.run(new, self.judging, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if gain > self.wobble:
            return GateDecision(True, gain, old_score, new_score,
                                reason="clearly better (gain > wobble)")
        if gain <= 0:
            return GateDecision(False, gain, old_score, new_score,
                                regressed=gain < 0,
                                reason="not better (gain <= 0)")
        return self._resolve_borderline(old, new, old_s, new_s, gain)

    def _resolve_borderline(self, old, new, old_s, new_s, gain) -> GateDecision:
        """Average the contested tasks over extra runs to see through the noise."""
        contested = [t for t in self.judging if old_s[t] != new_s[t]]
        old_acc = {t: [old_s[t]] for t in contested}
        new_acc = {t: [new_s[t]] for t in contested}
        for r in range(1, self.extra + 1):
            o = self.benchmark.run(old, contested, run_idx=r)
            n = self.benchmark.run(new, contested, run_idx=r)
            for t in contested:
                old_acc[t].append(o[t])
                new_acc[t].append(n[t])
        # Averaged gain over the whole judging set (non-contested contribute 0).
        diff = sum(
            sum(new_acc[t]) / len(new_acc[t]) - sum(old_acc[t]) / len(old_acc[t])
            for t in contested
        )
        avg_gain = diff / len(self.judging)
        accept = avg_gain > 0
        return GateDecision(
            accept, avg_gain, _mean(old_s), _mean(new_s),
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=f"borderline resolved: averaged gain {avg_gain:+.4f}",
        )
