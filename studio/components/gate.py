"""The gate (PRD §5.8): the referee, and the only thing that changes the harness.

It compares the harness old-way vs new-way on the judging set and decides
keep/reject with a **do-no-harm**, noise-aware rule:

  * gain  > wobble        -> clearly better -> accept
  * gain >= 0             -> do no harm     -> accept (no judging regression)
  * -wobble <= gain < 0   -> borderline     -> re-run the *contested* tasks a few
                              more times (capped) to average out noise, then
                              accept iff the averaged gain clears a tolerance.
  * gain < -wobble        -> regression     -> reject

``gain`` is the mean over judging tasks of the paired difference
``score(new) - score(old)``.

**Why do-no-harm, not must-improve.** Our original gate required *strict*
improvement (gain > 0) and rejected everything else. That is what lost to AHE:
AHE blind-commits every edit, so its harness accumulates additive surface (new
tools, skills, middleware) whose value shows up only on a broad held-out set —
inputs our capability-limited judging pool doesn't exercise, so they score
gain == 0 here and our must-improve gate threw them away. Worse, the Strategist
tends to *bundle* a system-prompt tweak with each additive edit, so even a
useful new tool arrives as a "behavioral" change that the strict gate rejects
when the prompt tweak doesn't move the pool.

Reframing never-regress as "never accept a *regression*" (gain >= 0) instead of
"only accept an *improvement*" (gain > 0) keeps our real differentiator — a
judging-set guard AHE lacks entirely — while letting the latent held-out value
accumulate. We are strictly more conservative than AHE (it accepts judging
regressions; we never do), just less timid than before.

**The additive nuance** (``additive=True`` — set when the edit does not touch
INSTRUCTIONS). Both additive and behavioral edits accept at gain >= 0; they
differ only inside the borderline band: a strictly additive edit (prose
untouched, so it can only *add* capability) gets a wider noise tolerance there,
while a behavioral edit (it rewrites guidance the agent already follows) must
average back to non-negative. The default is the stricter, behavioral path.

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
        if not self.judging:
            # No judging tasks = no evidence the edit is safe. Do-no-harm needs a
            # signal to clear; with none, stay conservative and reject.
            return GateDecision(False, 0.0, 0.0, 0.0,
                                reason="no judging tasks (no signal)")
        old_s = self.benchmark.run(old, self.judging, run_idx=0)
        new_s = self.benchmark.run(new, self.judging, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if gain > self.wobble:
            return GateDecision(True, gain, old_score, new_score,
                                reason="clearly better (gain > wobble)")
        # Do-no-harm: a non-regressing edit is kept, so latent held-out value
        # (additive surface the pool doesn't exercise) can accumulate like AHE's
        # blind-commit — but we never accept a measurable judging regression.
        if gain >= 0:
            kind = "additive" if additive else "behavioral"
            return GateDecision(True, gain, old_score, new_score,
                                reason=f"{kind}, does no harm (gain >= 0)")
        if gain < -self.wobble:
            return GateDecision(False, gain, old_score, new_score,
                                regressed=True,
                                reason="regresses past wobble")
        # -wobble <= gain < 0: borderline, average the contested tasks over the
        # noise before deciding.
        return self._resolve_borderline(old, new, old_s, new_s, gain,
                                        additive=additive)

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
            # Strictly additive: accept as long as the averaged result is not a
            # real regression. A tiny residual negative within noise still passes.
            tol = -self.wobble / 2
            accept = avg_gain >= tol
            reason = (f"additive borderline resolved: averaged gain "
                      f"{avg_gain:+.4f} (tol {tol:+.4f})")
        else:
            # Behavioral: it rewrote existing guidance, so require the noise-
            # averaged result back to do-no-harm (non-negative).
            accept = avg_gain >= 0
            reason = f"behavioral borderline resolved: averaged gain {avg_gain:+.4f}"
        return GateDecision(
            accept, avg_gain, _mean(old_s), _mean(new_s),
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=reason,
        )
