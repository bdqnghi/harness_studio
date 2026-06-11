"""The Orchestrator (PRD §5.12): the deterministic spine of both loops.

The **inner loop** (per round): find failures -> diagnose+blame -> propose several
competing strategies -> shell -> review -> rank -> structural check -> gate ->
snapshot. The gate is the only place an AI proposal becomes a harness mutation.

The **outer loop** (per segment of K rounds): the deep auditor re-checks the
harness on the big audit set (rewinding secret regressions, surfacing traps), then
the family map is updated — cheaply by rules, or by the Tier-A Meta-agent on a
plateau — revising *how* the next segment proposes. The family map is the shared
file joining the two loops.

The Orchestrator owns all state and validates that no AI component crosses the
gate boundary: the gate and deep auditor get a benchmark, never a Backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Backend
from .benchmark.base import Benchmark
from .benchmark.instrument import InstrumentedBenchmark, RewardHackError
from .components import (
    deep_auditor, diagnoser, health, mapper, meta_agent, ranker, reviewer,
    runner, shell, strategist, structural_check, wobble,
)
from .components.family_map import FamilyMap, init_map
from .components.gate import Gate
from .components.snapshotter import Snapshotter
from .components.splitter import TaskSplit, sample_practice, split_tasks
from .components.strategist import Strategy
from .config import Config
from .harness import Harness
from .parts import PartMap
from .state import RoundOutcome, WorkspaceState


@dataclass
class RunResult:
    baseline_final: float
    final_score: float
    wobble: float
    rounds: list[RoundOutcome] = field(default_factory=list)
    family_map: FamilyMap | None = None
    task_runs: int = 0  # task-score evaluations actually executed (cost)
    cache_hits: int = 0
    halted: bool = False  # set if a reward-hacking incident stopped the run

    @property
    def uplift(self) -> float:
        return self.final_score - self.baseline_final

    @property
    def accepted(self) -> int:
        return sum(1 for r in self.rounds if r.accepted)

    @property
    def cost_per_point(self) -> float:
        """Task runs per percentage-point of final-exam uplift (PRD §9.2).
        ``inf`` if there was no improvement."""
        pts = self.uplift * 100
        return self.task_runs / pts if pts > 0 else float("inf")


class Orchestrator:
    def __init__(
        self,
        *,
        workspace: Path,
        source_harness: Harness,
        benchmark: Benchmark,
        backend: Backend,
        config: Config,
        split: TaskSplit | None = None,
        part_map: PartMap | None = None,
    ) -> None:
        self.config = config
        # Wrap the benchmark: caching + cost counters + reward-hack defense.
        self.benchmark: Benchmark = InstrumentedBenchmark(benchmark, cache=config.cache)
        self.backend = backend  # only the Strategist gets this; never the gate
        self.state = WorkspaceState(root=Path(workspace))
        # Install the harness under the workspace as the live copy.
        self.harness = source_harness.copy_to(self.state.harness_dir)
        # An explicit split (e.g. from a test) overrides the computed one.
        self.split: TaskSplit = split or split_tasks(
            benchmark.list_tasks(), config.piles, seed=config.seed
        )
        # Fail fast on a misconfiguration that would silently do nothing: tasks
        # exist but the practice pool is empty (piles over-allocate the total).
        if self.benchmark.list_tasks() and not self.split.practice:
            raise ValueError(
                "practice pool is empty; reduce pile sizes relative to task count"
            )
        # An explicit part map (test/toy) overrides running the AI Mapper; when
        # the Mapper produced it, we re-map at each segment boundary.
        self._remap = part_map is None
        self.part_map: PartMap = part_map or mapper.map_harness(backend, self.harness)
        self.snapshotter = Snapshotter(self.state.snapshots_dir)

        # Outer-loop state.
        self.family_map: FamilyMap = init_map(self.state.family_map_path)
        self._segment_accepted: list[str] = []  # families accepted this segment
        self._best_audit_score: float | None = None
        self._best_dir = self.state.root / "best"

    # --- setup ---

    def _final_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.final_exam, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    def _judging_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.judging, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    def _audit_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.audit, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    # --- the two loops ---

    def run(self) -> RunResult:
        baseline_final = self._final_score()
        judging_wobble = wobble.measure_wobble(
            self.benchmark, self.harness, self.split.judging,
            runs=self.config.loop.wobble_runs,
        )
        regression_wobble = (
            wobble.measure_wobble(
                self.benchmark, self.harness, self.split.regression,
                runs=self.config.loop.wobble_runs,
            )
            if self.split.regression else 0.0
        )
        self.state.wobble = max(judging_wobble, regression_wobble)
        self.snapshotter.save(self.harness, 0, self._judging_score())
        # Seed the deep-audit "best so far" with the baseline harness.
        self._best_audit_score = self._audit_score()
        self.harness.copy_to(self._best_dir)

        total = self.config.loop.rounds
        seg_len = max(1, self.config.loop.segment_length)
        halted = False
        try:
            for r in range(1, total + 1):
                self._round(r)
                self._assess_health()
                if r % seg_len == 0 and r < total:  # segment boundary (not last)
                    self._segment_boundary(r)
            if total > 0:  # deep-audit the trailing segment the loop never closed
                self._finalize(total)
        except RewardHackError as e:
            self.state.health.reward_hack_incidents += 1
            self.state.health_log.append(f"HALT reward_hack: {e}")
            halted = True

        return RunResult(
            baseline_final=baseline_final,
            final_score=self._final_score(),
            wobble=self.state.wobble,
            rounds=list(self.state.evidence),
            family_map=self.family_map,
            task_runs=self.benchmark.task_runs,
            cache_hits=self.benchmark.cache_hits,
            halted=halted,
        )

    def _assess_health(self) -> None:
        for sig in health.assess(self.state.health, self.config.health):
            self.state.health_log.append(f"{sig.name}: {sig.detail} -> {sig.response}")

    # --- outer loop: the segment boundary (PRD §5.10, §5.11) ---

    def _audit_and_update(self, round_idx: int):
        """Deep-audit the segment, rewind a secret regression, and update the
        family map by rule. Returns (verdict, was_plateau). Resets the segment."""
        verdict = deep_auditor.audit(
            self.benchmark, self.harness, self.split.audit,
            best_score=self._best_audit_score, wobble=self.state.wobble,
        )
        traps: list[str] = []
        if verdict.verdict == "worse":
            # Secretly worse: rewind to the best harness; the families accepted
            # this segment are traps (passed fast gate, failed deep audit).
            self.harness = Harness(self._best_dir).copy_to(self.state.harness_dir)
            traps = list(dict.fromkeys(self._segment_accepted))
        elif verdict.verdict == "better":
            self._best_audit_score = verdict.score
            self.harness.copy_to(self._best_dir)

        survived = [f for f in self._segment_accepted if f not in traps]
        meta_agent.rule_based_update(self.family_map, survived, traps)
        was_plateau = not self._segment_accepted
        self._segment_accepted = []
        return verdict, was_plateau

    def _segment_boundary(self, round_idx: int) -> None:
        verdict, plateau = self._audit_and_update(round_idx)
        # Escalate to the Tier-A Meta-agent on a plateau (no accepted gains).
        if plateau:
            self._escalate_meta_agent(round_idx, verdict)
        self.family_map.save(self.state.family_map_path)
        if self._remap:  # codebase changed as edits landed; re-label it
            self.part_map = mapper.map_harness(self.backend, self.harness)

    def _finalize(self, round_idx: int) -> None:
        """Close the trailing segment the round loop never reached a boundary for:
        deep-audit + rewind + rule-based map update, but no meta escalation and no
        re-map (the run is ending). This guarantees the final number is audited."""
        self._audit_and_update(round_idx)
        self.family_map.save(self.state.family_map_path)

    def _escalate_meta_agent(self, round_idx: int, verdict) -> None:
        mech = self.state.root / "mechanism"
        mech.mkdir(parents=True, exist_ok=True)
        self.family_map.save(mech / "family_map.md")
        (mech / "segment_evidence.md").write_text(self._segment_evidence_md(round_idx, verdict))
        meta_agent.escalate(self.backend, mech)
        # Read back the (only) mechanism file the meta-agent may edit.
        self.family_map = FamilyMap.load(mech / "family_map.md")

    def _segment_evidence_md(self, round_idx: int, verdict) -> str:
        recent = [o for o in self.state.evidence if o.round_idx <= round_idx][-self.config.loop.segment_length:]
        lines = [
            f"# Segment ending at round {round_idx}", "",
            f"Deep-audit verdict: {verdict.verdict} (score {verdict.score:.3f})",
            f"Accepted families this segment: {self._segment_accepted or '(none — plateau)'}",
            f"Consecutive gate rejections: {self.state.health.gate_rejections}", "",
            "## Round outcomes", *(f"- round {o.round_idx}: "
              f"{'accept ' + o.family_label if o.accepted else 'reject'} — {o.note}"
              for o in recent),
        ]
        return "\n".join(lines) + "\n"

    def _round(self, round_idx: int) -> None:
        # 1. Find failures, then diagnose + blame.
        practice = sample_practice(
            self.split, self.config.piles.practice, self.config.seed, round_idx
        )
        report = runner.run_practice(self.benchmark, self.harness, practice)
        if not report.failures:
            self._reject(round_idx, "no failures on the practice batch")
            return
        diagnosis = diagnoser.diagnose(self.backend, report.failures)

        # 2. Propose several competing whole-strategies (each its own candidate).
        round_dir = self.state.candidates_dir / f"round_{round_idx:03d}"
        strategies = strategist.propose_many(
            self.backend, self.harness, round_dir, diagnosis,
            n=self.config.loop.strategies_per_round,
            id_prefix=f"r{round_idx}",
            do_not_touch=self.part_map.do_not_touch,
            family_map_text=self._family_map_text(),
        )

        # 3. Code shell on each: revert do-not-touch, enforce budget, label family.
        survivors = self._shell_filter(strategies)
        if not survivors:
            self._reject(round_idx, "all strategies dropped at the shell")
            return

        # 4. Review (prune known-dead / incoherent) then rank for testing order.
        survivors = self._review(survivors)
        if not survivors:
            self._reject(round_idx, "all strategies dropped at review")
            return
        survivors = self._rank(survivors)

        # 5. Structural check + gate, top-1 with fall-through (PRD §5.8, §11 Q3).
        self._test_in_order(round_idx, survivors)

    # --- round helpers ---

    def _shell_filter(self, strategies: list[Strategy]) -> list[Strategy]:
        survivors = []
        for s in strategies:
            res = shell.enforce(
                self.harness, s.candidate, self.part_map,
                budget_per_part=self.config.edits.budget_per_part,
            )
            if not res.ok:
                self.state.avoid_list.extend(res.violations)
                continue
            if not res.changed_parts:
                continue  # no editable-part change (empty / all reverted)
            s.changed_parts = res.changed_parts
            s.family_label = strategist.family_label(res.changed_parts)
            survivors.append(s)
        return survivors

    def _summaries(self, strategies: list[Strategy]) -> list[dict]:
        return [
            {
                "strategy_id": s.strategy_id,
                "family_label": s.family_label,
                "changed_parts": sorted(p.value for p in s.changed_parts),
                "intent": s.intent,
            }
            for s in strategies
        ]

    def _review(self, strategies: list[Strategy]) -> list[Strategy]:
        verdict = reviewer.review(
            self.backend, self._summaries(strategies), self._do_not_repeat()
        )
        keep = set(verdict.get("keep", []))
        dropped = {d["strategy_id"] for d in verdict.get("drop", [])}
        # Keep anything explicitly kept or not explicitly dropped (err toward keeping).
        return [s for s in strategies if s.strategy_id in keep or s.strategy_id not in dropped]

    def _rank(self, strategies: list[Strategy]) -> list[Strategy]:
        order = ranker.rank(self.backend, self._summaries(strategies))
        by_id = {s.strategy_id: s for s in strategies}
        return [by_id[i] for i in order if i in by_id]

    def _test_in_order(self, round_idx: int, strategies: list[Strategy]) -> None:
        gate = Gate(
            self.benchmark, self.split.judging, self.state.wobble,
            regression_tasks=self.split.regression,  # dual-split when populated (choose_split); [] -> single-split
            borderline_extra_runs=self.config.gate.borderline_extra_runs,
        )
        last_note = "no strategy passed the gate"
        for s in strategies:
            struct = structural_check.check(
                s.candidate, self.benchmark, backend=self.backend,
                do_not_touch=self.part_map.do_not_touch,
                allow_repair=self.config.edits.allow_repair,
            )
            if not struct.ok:
                self.state.avoid_list.append(struct.error)
                last_note = f"structural check failed: {struct.error}"
                continue
            if struct.repaired:
                # The repair agent edited files; re-enforce the shell invariants
                # (it could have touched a do-not-touch file or blown the budget).
                res = shell.enforce(
                    self.harness, s.candidate, self.part_map,
                    budget_per_part=self.config.edits.budget_per_part,
                )
                if not res.ok or not res.changed_parts:
                    last_note = "repair violated shell invariants"
                    continue
            # Only a pure file addition is structurally additive. Rewriting or
            # deleting any existing file can alter behavior the visible gate does
            # not exercise, regardless of which part type owns that file.
            additive = shell.is_strictly_additive(self.harness, s.candidate)
            decision = gate.evaluate(self.harness, s.candidate, additive=additive)
            if decision.accept:
                self.harness = s.candidate.copy_to(self.state.harness_dir)
                self.state.health.gate_rejections = 0
                self.state.health.empty_rounds = 0
                self._segment_accepted.append(s.family_label)
                self.state.record(RoundOutcome(
                    round_idx, True, decision.gain, decision.old_score,
                    decision.new_score, family_label=s.family_label,
                    note=f"{s.strategy_id} accepted: {decision.reason}",
                ))
                self.snapshotter.save(self.harness, round_idx, self._judging_score())
                return
            last_note = f"{s.strategy_id} rejected: {decision.reason}"
        # Strategies were produced and tested but none passed the gate: this is a
        # gate-rejection round, not an "empty" one (PRD §7 keeps the signals apart).
        self.state.health.gate_rejections += 1
        self.state.health.empty_rounds = 0
        self._reject(round_idx, last_note, empty=False)

    def _reject(self, round_idx: int, note: str, *, empty: bool = True) -> None:
        if empty:  # nothing testable was produced (dropped at shell/review, or no failures)
            self.state.health.empty_rounds += 1
        score = self._judging_score()
        self.state.record(
            RoundOutcome(round_idx, False, 0.0, score, score, note=note)
        )
        self.snapshotter.save(self.harness, round_idx, score)

    # --- mechanism state read by the inner loop each round ---

    def _family_map_text(self) -> str:
        return self.family_map.to_text()

    def _do_not_repeat(self) -> list[str]:
        return self.family_map.do_not_repeat()
