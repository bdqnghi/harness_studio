"""The gate (PRD §5.8): the referee, and the only thing that changes the harness.

It compares the harness old-way vs new-way on the judging set and decides
keep/reject with a **do-no-harm**, noise-aware rule:

  * gain  > wobble        -> clearly better -> accept
  * -wobble <= gain <= wobble -> borderline -> re-run the full predeclared split
                                  a few more times (capped) to average out noise,
                                  then require a behavioral edit to clear the
                                  residual noise threshold.
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

**The additive nuance** (``additive=True`` — set only when the diff exclusively
adds files). A pure addition may be neutral on visible tasks. Any modification or
deletion is behavioral and must demonstrate a gain beyond residual noise.

Protection (PRD §3): the gate is constructed with a benchmark only — never a
Backend. No AI helper can reach the evaluator or write scores.
"""

from __future__ import annotations

import math
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
    regression_gain: float = 0.0  # dual-split: gain on the regression set (0 in single-split mode)


def _mean(d: dict[str, float]) -> float:
    return sum(d.values()) / len(d) if d else 0.0


class Gate:
    def __init__(
        self,
        benchmark: Benchmark,
        judging_tasks: list[str],
        wobble: float,
        *,
        regression_tasks: list[str] | None = None,
        borderline_extra_runs: int = 5,
        aggregate_accept: bool = False,
    ) -> None:
        self.benchmark = benchmark
        self.judging = list(judging_tasks)
        # Dual-split (Self-Harness): a disjoint regression set checked at every
        # accept. None/empty -> single-split do-no-harm (legacy path).
        self.regression = list(regression_tasks or [])
        self.wobble = max(0.0, wobble)
        self.extra = max(0, borderline_extra_runs)
        # aggregate_accept (off by default): score the edit on the POOLED held-in
        # set (judging ∪ regression) and accept on the pooled gain, instead of
        # requiring EACH slice to independently not-regress. The strict dual
        # split rejects an edit that genuinely helps overall when it lands on
        # tasks in one slice and the other slice wobbles negative from noise —
        # a real failure mode on noisy benchmarks (observed on tau2 cold-start).
        self.aggregate_accept = aggregate_accept

    def evaluate(self, old: Harness, new: Harness, *, additive: bool = False) -> GateDecision:
        if self.regression and self.aggregate_accept:
            return self._evaluate_aggregate(old, new, additive=additive)
        if self.regression:
            return self._evaluate_dual(old, new, additive=additive)
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
        if gain < -self.wobble:
            return GateDecision(False, gain, old_score, new_score,
                                regressed=True,
                                reason="regresses past wobble")
        if self.wobble == 0:
            if additive or gain > 0:
                kind = "additive" if additive else "behavioral"
                return GateDecision(True, gain, old_score, new_score,
                                    reason=f"{kind}, exact non-regression")
            return GateDecision(False, gain, old_score, new_score,
                                reason="behavioral edit has no strict gain")
        # Any result inside [-wobble, +wobble] is unresolved, including a small
        # positive result. Average the full split before deciding.
        return self._resolve_borderline(old, new, old_s, new_s, gain,
                                        additive=additive)

    def _resolve_borderline(self, old, new, old_s, new_s, gain,
                            *, additive: bool = False) -> GateDecision:
        """Average the full predeclared split over extra runs."""
        old_acc = {t: [old_s[t]] for t in self.judging}
        new_acc = {t: [new_s[t]] for t in self.judging}
        for r in range(1, self.extra + 1):
            o = self.benchmark.run(old, self.judging, run_idx=r)
            n = self.benchmark.run(new, self.judging, run_idx=r)
            for t in self.judging:
                old_acc[t].append(o[t])
                new_acc[t].append(n[t])
        diff = sum(
            sum(new_acc[t]) / len(new_acc[t]) - sum(old_acc[t]) / len(old_acc[t])
            for t in self.judging
        )
        avg_gain = diff / len(self.judging)
        residual_wobble = self.wobble / math.sqrt(1 + self.extra)
        if additive:
            # Strictly additive: accept as long as the averaged result is not a
            # real regression. A tiny residual negative within noise still passes.
            tol = -residual_wobble
            accept = avg_gain >= tol
            reason = (f"additive borderline resolved: averaged gain "
                      f"{avg_gain:+.4f} (tol {tol:+.4f})")
        else:
            # Behavioral edits must clear the residual noise after re-runs.
            accept = avg_gain > residual_wobble
            reason = (
                f"behavioral borderline resolved: averaged gain {avg_gain:+.4f} "
                f"(must exceed {residual_wobble:+.4f})"
            )
        return GateDecision(
            accept, avg_gain, _mean(old_s), _mean(new_s),
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=reason,
        )

    # --- aggregate-accept (pooled held-in gain; opt-in) -------------------------

    def _evaluate_aggregate(self, old, new, *, additive: bool) -> GateDecision:
        """Do-no-harm on the POOLED held-in set (judging ∪ regression).

        Accept iff the edit improves the pooled set beyond the noise floor (or,
        if additive, doesn't regress it). This captures edits that help overall
        even when the gain concentrates in one slice and the other wobbles
        negative — which the strict per-slice dual gate rejects. The pooled mean
        is a larger sample, so it is also less noisy than either slice alone."""
        tasks = list(dict.fromkeys(self.judging + self.regression))
        # regression-slice gain is still computed for the decision log/field.
        gr, _, _ = self._gain_on(old, new, self.regression) if self.regression else (0.0, {}, {})
        old_s = self.benchmark.run(old, tasks, run_idx=0)
        new_s = self.benchmark.run(new, tasks, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if gain > self.wobble:
            return GateDecision(True, gain, old_score, new_score, regression_gain=gr,
                                reason=f"aggregate gain {gain:+.3f} > wobble (pooled {len(tasks)})")
        if gain < -self.wobble:
            return GateDecision(False, gain, old_score, new_score, regression_gain=gr,
                                regressed=True,
                                reason=f"aggregate regresses ({gain:+.3f} pooled)")
        if self.wobble == 0:
            if additive or gain > 0:
                return GateDecision(True, gain, old_score, new_score, regression_gain=gr,
                                    reason=f"aggregate exact non-regression ({gain:+.3f})")
            return GateDecision(False, gain, old_score, new_score, regression_gain=gr,
                                reason="aggregate behavioral edit has no strict gain")
        # Borderline: average the pooled set over extra runs, then decide.
        old_acc = {t: [old_s[t]] for t in tasks}
        new_acc = {t: [new_s[t]] for t in tasks}
        for r in range(1, self.extra + 1):
            o = self.benchmark.run(old, tasks, run_idx=r)
            n = self.benchmark.run(new, tasks, run_idx=r)
            for t in tasks:
                old_acc[t].append(o[t]); new_acc[t].append(n[t])
        avg_gain = sum(
            sum(new_acc[t]) / len(new_acc[t]) - sum(old_acc[t]) / len(old_acc[t])
            for t in tasks
        ) / len(tasks)
        residual = self.wobble / math.sqrt(1 + self.extra)
        accept = (avg_gain >= -residual) if additive else (avg_gain > residual)
        return GateDecision(
            accept, avg_gain, old_score, new_score, regression_gain=gr,
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=f"aggregate borderline resolved: pooled gain {avg_gain:+.4f} "
                   f"(threshold {residual:+.4f})",
        )

    # --- dual-split (Self-Harness 2606.09498) -----------------------------------

    def _gain_on(self, old, new, tasks, *, run_idx=0):
        os_ = self.benchmark.run(old, tasks, run_idx=run_idx)
        ns_ = self.benchmark.run(new, tasks, run_idx=run_idx)
        return _mean(ns_) - _mean(os_), os_, ns_

    def _accepts(
        self, gj: float, gr: float, *, additive: bool, tol: float = 0.0,
        improve: float = 0.0,
    ) -> bool:
        """The dual-split acceptance rule.

        Both splits must be non-regressing (within ``tol``). A behavioral edit
        (it rewrites existing guidance) must ALSO strictly improve at least one
        split — `max>0` (Self-Harness C2), so we don't accumulate neutral prose
        churn. A strictly-additive edit may be neutral-on-both: it only *adds*
        capability, so do-no-harm is enough to keep the latent surface."""
        non_regress = (gj >= tol) and (gr >= tol)
        if not non_regress:
            return False
        if additive:
            return True
        return max(gj, gr) > improve

    def _evaluate_dual(self, old, new, *, additive: bool) -> GateDecision:
        if not self.judging or not self.regression:
            return GateDecision(False, 0.0, 0.0, 0.0,
                                reason="dual-split needs both judging and regression")
        gj, oj, nj = self._gain_on(old, new, self.judging)
        gr, o_r, n_r = self._gain_on(old, new, self.regression)
        old_score, new_score = _mean(oj), _mean(nj)
        gmin = min(gj, gr)

        # Clear regression on either split -> reject.
        if gmin < -self.wobble:
            return GateDecision(False, gj, old_score, new_score, regression_gain=gr,
                                regressed=True,
                                reason=f"regresses a split (judging {gj:+.3f}, regression {gr:+.3f})")
        # Pure additions may be neutral on both visible splits.
        if additive and gmin >= 0:
            return GateDecision(True, gj, old_score, new_score, regression_gain=gr,
                                reason=f"dual-split ok (additive; judging {gj:+.3f}, regression {gr:+.3f})")
        # A behavioral edit is clear only when its improvement beats wobble.
        if gmin >= 0 and max(gj, gr) > self.wobble:
            if self._accepts(gj, gr, additive=additive):
                kind = "additive" if additive else "behavioral"
                return GateDecision(True, gj, old_score, new_score, regression_gain=gr,
                                    reason=f"dual-split ok ({kind}; judging {gj:+.3f}, regression {gr:+.3f})")
        if self.wobble == 0:
            return GateDecision(False, gj, old_score, new_score, regression_gain=gr,
                                reason=f"no strict gain (judging {gj:+.3f}, regression {gr:+.3f})")
        # Borderline: at least one result lies inside the noise band.
        return self._resolve_dual_borderline(old, new, oj, nj, o_r, n_r, gj, gr, additive=additive)

    def _resolve_dual_borderline(self, old, new, oj, nj, o_r, n_r, gj, gr, *, additive) -> GateDecision:
        def avg_gain(tasks, o0, n0):
            oacc = {t: [o0[t]] for t in tasks}
            nacc = {t: [n0[t]] for t in tasks}
            for r in range(1, self.extra + 1):
                o = self.benchmark.run(old, tasks, run_idx=r)
                n = self.benchmark.run(new, tasks, run_idx=r)
                for t in tasks:
                    oacc[t].append(o[t]); nacc[t].append(n[t])
            diff = sum(
                sum(nacc[t]) / len(nacc[t]) - sum(oacc[t]) / len(oacc[t])
                for t in tasks
            )
            return diff / len(tasks)

        aj = avg_gain(self.judging, oj, nj)
        ar = avg_gain(self.regression, o_r, n_r)
        residual_wobble = self.wobble / math.sqrt(1 + self.extra)
        tol = -residual_wobble if additive else 0.0
        improve = 0.0 if additive else residual_wobble
        accept = self._accepts(
            aj, ar, additive=additive, tol=tol, improve=improve
        )
        return GateDecision(
            accept, aj, _mean(oj), _mean(nj), regression_gain=ar,
            regressed=min(aj, ar) < 0, borderline=True, runs_used=1 + self.extra,
            reason=f"dual-split borderline resolved (judging {aj:+.3f}, regression {ar:+.3f}, tol {tol:+.3f})",
        )
