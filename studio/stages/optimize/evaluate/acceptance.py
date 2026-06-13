"""The acceptance (PRD §5.8): the referee, and the only thing that changes the harness.

It compares the harness old-way vs new-way on the judging set and decides
keep/reject with a **do-no-harm**, noise-aware rule:

  * gain  > noise_floor        -> clearly better -> accept
  * -noise_floor <= gain <= noise_floor -> borderline -> re-run the full predeclared split
                                  a few more times (capped) to average out noise,
                                  then require a behavioral edit to clear the
                                  residual noise threshold.
  * gain < -noise_floor        -> regression     -> reject

``gain`` is the mean over judging tasks of the paired difference
``score(new) - score(old)``.

**Why do-no-harm, not must-improve.** Our original acceptance required *strict*
improvement (gain > 0) and rejected everything else. That is what lost to AHE:
AHE blind-commits every edit, so its harness accumulates additive surface (new
tools, skills, middleware) whose value shows up only on a broad held-out set —
inputs our capability-limited judging pool doesn't exercise, so they score
gain == 0 here and our must-improve acceptance threw them away. Worse, the Strategist
tends to *bundle* a system-prompt tweak with each additive edit, so even a
useful new tool arrives as a "behavioral" change that the strict acceptance rejects
when the prompt tweak doesn't move the pool.

Reframing never-regress as "never accept a *regression*" (gain >= 0) instead of
"only accept an *improvement*" (gain > 0) keeps our real differentiator — a
judging-set guard AHE lacks entirely — while letting the latent held-out value
accumulate. We are strictly more conservative than AHE (it accepts judging
regressions; we never do), just less timid than before.

**The additive nuance** (``additive=True`` — set only when the diff exclusively
adds files). A pure addition may be neutral on visible tasks. Any modification or
deletion is behavioral and must demonstrate a gain beyond residual noise.

Protection (PRD §3): the acceptance is constructed with a benchmark only — never a
Backend. No AI helper can reach the evaluator or write scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from studio.benchmark.base import Benchmark
from studio.core.harness import Harness


@dataclass
class AcceptanceDecision:
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


class AcceptanceCheck:
    def __init__(
        self,
        benchmark: Benchmark,
        judging_tasks: list[str],
        noise_floor: float,
        *,
        regression_tasks: list[str] | None = None,
        borderline_extra_runs: int = 5,
        strict_dual: bool = False,
    ) -> None:
        self.benchmark = benchmark
        self.judging = list(judging_tasks)
        # A disjoint do-no-harm set, pooled with the held-in (judging) set into
        # the NET decision. None/empty -> single-split do-no-harm.
        self.regression = list(regression_tasks or [])
        self.noise_floor = max(0.0, noise_floor)
        self.extra = max(0, borderline_extra_runs)
        # Default: accept on the NET pooled gain over held_in ∪ regression, so a
        # real overall lift is kept even when one slice dips a little within
        # noise. Measurement variance dominates here, so a small regression must
        # not veto a genuine net improvement. ``strict_dual`` opts into the
        # stricter Self-Harness rule (each slice must independently not-regress)
        # for the rare case where ANY regression is unacceptable.
        self.strict_dual = strict_dual

    def evaluate(self, old: Harness, new: Harness, *, additive: bool = False) -> AcceptanceDecision:
        if self.regression and self.strict_dual:
            return self._evaluate_dual(old, new, additive=additive)
        if self.regression:
            return self._evaluate_aggregate(old, new, additive=additive)
        if not self.judging:
            # No judging tasks = no evidence the edit is safe. Do-no-harm needs a
            # signal to clear; with none, stay conservative and reject.
            return AcceptanceDecision(False, 0.0, 0.0, 0.0,
                                reason="no judging tasks (no signal)")
        old_s = self.benchmark.run(old, self.judging, run_idx=0)
        new_s = self.benchmark.run(new, self.judging, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if gain > self.noise_floor:
            return AcceptanceDecision(True, gain, old_score, new_score,
                                reason="clearly better (gain > noise_floor)")
        if gain < -self.noise_floor:
            return AcceptanceDecision(False, gain, old_score, new_score,
                                regressed=True,
                                reason="regresses past noise_floor")
        if self.noise_floor == 0:
            if additive or gain > 0:
                kind = "additive" if additive else "behavioral"
                return AcceptanceDecision(True, gain, old_score, new_score,
                                    reason=f"{kind}, exact non-regression")
            return AcceptanceDecision(False, gain, old_score, new_score,
                                reason="behavioral edit has no strict gain")
        # Any result inside [-noise_floor, +noise_floor] is unresolved, including a small
        # positive result. Average the full split before deciding.
        return self._resolve_borderline(old, new, old_s, new_s, gain,
                                        additive=additive)

    def _resolve_borderline(self, old, new, old_s, new_s, gain,
                            *, additive: bool = False) -> AcceptanceDecision:
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
        residual_noise_floor = self.noise_floor / math.sqrt(1 + self.extra)
        if additive:
            # Strictly additive: accept as long as the averaged result is not a
            # real regression. A tiny residual negative within noise still passes.
            tol = -residual_noise_floor
            accept = avg_gain >= tol
            reason = (f"additive borderline resolved: averaged gain "
                      f"{avg_gain:+.4f} (tol {tol:+.4f})")
        else:
            # Behavioral edits must clear the residual noise after re-runs.
            accept = avg_gain > residual_noise_floor
            reason = (
                f"behavioral borderline resolved: averaged gain {avg_gain:+.4f} "
                f"(must exceed {residual_noise_floor:+.4f})"
            )
        return AcceptanceDecision(
            accept, avg_gain, _mean(old_s), _mean(new_s),
            regressed=avg_gain < 0, borderline=True, runs_used=1 + self.extra,
            reason=reason,
        )

    # --- aggregate-accept (pooled held-in gain; opt-in) -------------------------

    def _evaluate_aggregate(self, old, new, *, additive: bool) -> AcceptanceDecision:
        """Do-no-harm on the POOLED held-in set (judging ∪ regression).

        Accept iff the edit improves the pooled set beyond the noise floor (or,
        if additive, doesn't regress it). This captures edits that help overall
        even when the gain concentrates in one slice and the other noise_floors
        negative — which the strict per-slice dual acceptance rejects. The pooled mean
        is a larger sample, so it is also less noisy than either slice alone."""
        tasks = list(dict.fromkeys(self.judging + self.regression))
        # regression-slice gain is still computed for the decision log/field.
        gr, _, _ = self._gain_on(old, new, self.regression) if self.regression else (0.0, {}, {})
        old_s = self.benchmark.run(old, tasks, run_idx=0)
        new_s = self.benchmark.run(new, tasks, run_idx=0)
        gain = _mean(new_s) - _mean(old_s)
        old_score, new_score = _mean(old_s), _mean(new_s)

        if gain > self.noise_floor:
            return AcceptanceDecision(True, gain, old_score, new_score, regression_gain=gr,
                                reason=f"aggregate gain {gain:+.3f} > noise_floor (pooled {len(tasks)})")
        if gain < -self.noise_floor:
            return AcceptanceDecision(False, gain, old_score, new_score, regression_gain=gr,
                                regressed=True,
                                reason=f"aggregate regresses ({gain:+.3f} pooled)")
        if self.noise_floor == 0:
            if additive or gain > 0:
                return AcceptanceDecision(True, gain, old_score, new_score, regression_gain=gr,
                                    reason=f"aggregate exact non-regression ({gain:+.3f})")
            return AcceptanceDecision(False, gain, old_score, new_score, regression_gain=gr,
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
        residual = self.noise_floor / math.sqrt(1 + self.extra)
        accept = (avg_gain >= -residual) if additive else (avg_gain > residual)
        return AcceptanceDecision(
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

    def _evaluate_dual(self, old, new, *, additive: bool) -> AcceptanceDecision:
        if not self.judging or not self.regression:
            return AcceptanceDecision(False, 0.0, 0.0, 0.0,
                                reason="dual-split needs both judging and regression")
        gj, oj, nj = self._gain_on(old, new, self.judging)
        gr, o_r, n_r = self._gain_on(old, new, self.regression)
        old_score, new_score = _mean(oj), _mean(nj)
        gmin = min(gj, gr)

        # Clear regression on either split -> reject.
        if gmin < -self.noise_floor:
            return AcceptanceDecision(False, gj, old_score, new_score, regression_gain=gr,
                                regressed=True,
                                reason=f"regresses a split (judging {gj:+.3f}, regression {gr:+.3f})")
        # Pure additions may be neutral on both visible splits.
        if additive and gmin >= 0:
            return AcceptanceDecision(True, gj, old_score, new_score, regression_gain=gr,
                                reason=f"dual-split ok (additive; judging {gj:+.3f}, regression {gr:+.3f})")
        # A behavioral edit is clear only when its improvement beats noise_floor.
        if gmin >= 0 and max(gj, gr) > self.noise_floor:
            if self._accepts(gj, gr, additive=additive):
                kind = "additive" if additive else "behavioral"
                return AcceptanceDecision(True, gj, old_score, new_score, regression_gain=gr,
                                    reason=f"dual-split ok ({kind}; judging {gj:+.3f}, regression {gr:+.3f})")
        if self.noise_floor == 0:
            return AcceptanceDecision(False, gj, old_score, new_score, regression_gain=gr,
                                reason=f"no strict gain (judging {gj:+.3f}, regression {gr:+.3f})")
        # Borderline: at least one result lies inside the noise band.
        return self._resolve_dual_borderline(old, new, oj, nj, o_r, n_r, gj, gr, additive=additive)

    def _resolve_dual_borderline(self, old, new, oj, nj, o_r, n_r, gj, gr, *, additive) -> AcceptanceDecision:
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
        residual_noise_floor = self.noise_floor / math.sqrt(1 + self.extra)
        tol = -residual_noise_floor if additive else 0.0
        improve = 0.0 if additive else residual_noise_floor
        accept = self._accepts(
            aj, ar, additive=additive, tol=tol, improve=improve
        )
        return AcceptanceDecision(
            accept, aj, _mean(oj), _mean(nj), regression_gain=ar,
            regressed=min(aj, ar) < 0, borderline=True, runs_used=1 + self.extra,
            reason=f"dual-split borderline resolved (judging {aj:+.3f}, regression {ar:+.3f}, tol {tol:+.3f})",
        )
