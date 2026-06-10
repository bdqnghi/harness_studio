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

**Do-no-harm acceptance for additive edits** (``additive=True``).
A behavioral edit (it rewrites the system prompt / INSTRUCTIONS) *replaces*
existing guidance, so the burden of proof is on it: it must measurably improve
the judging set or we reject it. But a *strictly additive* edit — it adds a new
tool, a middleware, a skill, a memory file without touching the prose the agent
already follows — can only help on inputs that exercise the new surface, and our
judging pool may simply not contain those inputs. Holding additive edits to
"must improve a capability-limited pool" throws away latent value that shows up
on a broader held-out set (this is exactly how AHE's blind-commit beat our
never-regress gate). So for additive edits we flip the burden: accept unless the
edit *regresses* beyond the noise floor.

  * gain >= 0             -> accept (do no harm; pool just doesn't exercise it)
  * -wobble <= gain < 0   -> borderline -> re-run contested, accept iff averaged
                              gain is within noise (>= a small negative tolerance)
  * gain < -wobble        -> reject (a genuine regression — additive or not)

The behavioral path (``additive=False``, the default) is unchanged.

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

    def evaluate(self, old: Harness, new: Harness, *, additive: bool = False) -> GateDecision:
        old_s = self.benchmark.run(old, self.judging, run_idx=0)
        new_s = self.benchmark.run(new, self.judging, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if additive:
            # Flip the burden of proof: accept unless it regresses past noise.
            if gain >= 0:
                return GateDecision(True, gain, old_score, new_score,
                                    reason="additive, does no harm (gain >= 0)")
            if gain < -self.wobble:
                return GateDecision(False, gain, old_score, new_score,
                                    regressed=True,
                                    reason="additive but regresses past wobble")
            # -wobble <= gain < 0: borderline, average out the noise.
            return self._resolve_borderline(old, new, old_s, new_s, gain,
                                            additive=True)

        if gain > self.wobble:
            return GateDecision(True, gain, old_score, new_score,
                                reason="clearly better (gain > wobble)")
        if gain <= 0:
            return GateDecision(False, gain, old_score, new_score,
                                regressed=gain < 0,
                                reason="not better (gain <= 0)")
        return self._resolve_borderline(old, new, old_s, new_s, gain)

    def _resolve_borderline(self, old, new, old_s, new_s, gain,
                            *, additive: bool = False) -> GateDecision:
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
        if additive:
            # Do-no-harm: accept as long as the averaged result is not a real
            # regression. A tiny residual negative within noise still passes.
            tol = -self.wobble / 2
            accept = avg_gain >= tol
            reason = (f"additive borderline resolved: averaged gain "
                      f"{avg_gain:+.4f} (tol {tol:+.4f})")
        else:
            accept = avg_gain > 0
            reason = f"borderline resolved: averaged gain {avg_gain:+.4f}"
        return GateDecision(
            accept, avg_gain, _mean(old_s), _mean(new_s),
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=reason,
        )
