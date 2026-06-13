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

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import Backend
from .benchmark.base import Benchmark
from .benchmark.instrument import InstrumentedBenchmark, RewardHackError
from .components import (
    deep_auditor, diagnoser, health, ideator, insight, localizer, mapper,
    meta_agent, ranker, reviewer, runner, shell, strategist, structural_check,
    wobble,
)
from .components.family_map import FamilyMap, init_map
from .components.gate import Gate
from .components.idea_tree import IdeaTree, classify_rejection, mutation_event
from .components.snapshotter import Snapshotter
from .components.splitter import TaskSplit, sample_held_in, split_tasks
from .components.strategist import Strategy
from .config import Config
from .harness import Harness
from .observe import ProgressLog, decision_dict
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
        self.benchmark: Benchmark = InstrumentedBenchmark(
            benchmark, cache=config.cache,
            disk_path=Path(config.score_cache) if config.score_cache else None,
        )
        self.backend = backend  # only the Strategist gets this; never the gate
        self.state = WorkspaceState(root=Path(workspace))
        # Install the harness under the workspace as the live copy.
        self.harness = source_harness.copy_to(self.state.harness_dir)
        # An explicit split (e.g. from a test) overrides the computed one.
        self.split: TaskSplit = split or split_tasks(
            benchmark.list_tasks(), config.piles, seed=config.seed
        )
        # Fail fast on a misconfiguration that would silently do nothing: tasks
        # exist but the held-in pool is empty (piles over-allocate the total).
        if self.benchmark.list_tasks() and not self.split.held_in:
            raise ValueError(
                "held-in pool is empty; reduce pile sizes relative to task count"
            )
        # An explicit part map (test/toy) overrides running the AI Mapper; when
        # the Mapper produced it, we re-map at each segment boundary.
        self._remap = part_map is None
        self.part_map: PartMap = part_map or mapper.map_harness(backend, self.harness)
        self.snapshotter = Snapshotter(self.state.snapshots_dir)
        self.progress = ProgressLog(self.state.progress_path)

        # Outer-loop state.
        self.family_map: FamilyMap = init_map(self.state.family_map_path)
        self._segment_accepted: list[str] = []  # families accepted this segment
        self._best_audit_score: float | None = None
        self._best_dir = self.state.root / "best"

        # The tree optimizer (A/B treatment arm). The classic arm must never
        # create tree state — the flag fully selects the round implementation.
        if config.loop.optimizer not in ("classic", "tree"):
            raise ValueError(f"unknown optimizer {config.loop.optimizer!r}")
        self.tree: IdeaTree | None = None
        self._segment_accepted_nodes: list[str] = []  # tree: node ids this segment
        if config.loop.optimizer == "tree":
            self.tree = IdeaTree.load_or_create(
                self.state.root / "idea_tree.json",
                md_path=self.state.root / "tree.md",
            )

    # --- setup ---

    def _final_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.held_out, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    def _held_in_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.held_in, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    def _audit_score(self) -> float:
        scores = self.benchmark.run(self.harness, self.split.held_in, run_idx=0)
        return sum(scores.values()) / len(scores) if scores else 0.0

    # --- the two loops ---

    def run(self) -> RunResult:
        self.progress.emit(
            "run_start",
            optimizer=getattr(self.config.loop, "optimizer", "classic"),
            rounds=self.config.loop.rounds,
            segment_length=self.config.loop.segment_length,
            n_held_in=len(self.split.held_in),
            n_regression=len(self.split.regression),
            n_held_out=len(self.split.held_out),
        )
        baseline_final = self._final_score()
        held_in_wobble = wobble.measure_wobble(
            self.benchmark, self.harness, self.split.held_in,
            runs=self.config.loop.wobble_runs,
        )
        regression_wobble = (
            wobble.measure_wobble(
                self.benchmark, self.harness, self.split.regression,
                runs=self.config.loop.wobble_runs,
            )
            if self.split.regression else 0.0
        )
        self.state.wobble = max(held_in_wobble, regression_wobble)
        self.snapshotter.save(self.harness, 0, self._held_in_score())
        # Seed the deep-audit "best so far" with the baseline harness.
        self._best_audit_score = self._audit_score()
        self.harness.copy_to(self._best_dir)
        self.progress.emit("setup_done", wobble=round(self.state.wobble, 4),
                           task_runs=self.benchmark.task_runs)

        total = self.config.loop.rounds
        seg_len = max(1, self.config.loop.segment_length)
        tree_mode = self.config.loop.optimizer == "tree"
        halted = False
        try:
            for r in range(1, total + 1):
                self.progress.emit("round_start", round=r)
                started = time.monotonic()
                if tree_mode:
                    self._round_tree(r)
                else:
                    self._round(r)
                outcome = self.state.evidence[-1] if self.state.evidence else None
                fields = outcome.to_dict() if outcome else {}
                fields.pop("round", None)  # the explicit round kwarg wins
                self.progress.emit(
                    "round_end", round=r,
                    wall_sec=round(time.monotonic() - started, 1),
                    task_runs=self.benchmark.task_runs,
                    cache_hits=self.benchmark.cache_hits,
                    **fields,
                )
                self._assess_health()
                if r % seg_len == 0 and r < total:  # segment boundary (not last)
                    if tree_mode:
                        self._segment_boundary_tree(r)
                    else:
                        self._segment_boundary(r)
            if total > 0:  # deep-audit the trailing segment the loop never closed
                if tree_mode:
                    self._finalize_tree(total)
                else:
                    self._finalize(total)
        except RewardHackError as e:
            self.state.health.reward_hack_incidents += 1
            self.state.log_health(f"HALT reward_hack: {e}")
            self.progress.emit("halt", reason="reward_hack", detail=str(e))
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
            self.state.log_health(f"{sig.name}: {sig.detail} -> {sig.response}")
            self.progress.emit("health_signal", name=sig.name, detail=sig.detail,
                               response=sig.response)

    # --- outer loop: the segment boundary (PRD §5.10, §5.11) ---

    def _audit_and_update(self, round_idx: int):
        """Deep-audit the segment, rewind a secret regression, and update the
        family map by rule. Returns (verdict, was_plateau). Resets the segment."""
        verdict = deep_auditor.audit(
            self.benchmark, self.harness, self.split.held_in,
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
        self.progress.emit("segment_boundary", round=round_idx,
                           audit_verdict=verdict.verdict,
                           audit_score=round(verdict.score, 4), plateau=plateau)
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
        verdict, plateau = self._audit_and_update(round_idx)
        self.progress.emit("segment_boundary", round=round_idx, final=True,
                           audit_verdict=verdict.verdict,
                           audit_score=round(verdict.score, 4), plateau=plateau)
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

    def _localize(self, round_idx: int, patterns: list[dict], round_dir):
        """Materialize this round's failure evidence and run the localizer.

        Returns ``(localization_targets, evidence_dir)``. Off → ([], None). The
        evidence is versioned to the LIVE harness hash (the candidate doesn't
        exist yet), and localization is a hint: any failure degrades to []."""
        if self.config.loop.localizer == "off":
            return [], None
        store = getattr(self.benchmark, "evidence_store", None)
        if store is None:
            self.progress.emit("localization_done", round=round_idx, n_targets=0,
                               mode=self.config.loop.localizer, reason="no evidence store")
            return [], None
        try:
            evidence_dir = store.materialize(
                self.harness.content_hash(), Path(round_dir) / "evidence")
        except Exception:  # noqa: BLE001 — never break a round on evidence I/O
            return [], None
        localization = localizer.localize(
            self.backend, patterns, self.harness, evidence_dir,
            editable_files=self.part_map.editable_files(),
            mode=self.config.loop.localizer,
        )
        self.progress.emit("localization_done", round=round_idx,
                           n_targets=len(localization), mode=self.config.loop.localizer)
        return localization, evidence_dir

    def _round(self, round_idx: int) -> None:
        # 1. Find failures, then diagnose + blame.
        batch = sample_held_in(
            self.split, self.config.piles.round_size, self.config.seed, round_idx
        )
        report = runner.run_batch(self.benchmark, self.harness, batch)
        self.progress.emit("batch_done", round=round_idx,
                           pass_rate=round(report.pass_rate, 4),
                           n_failures=len(report.failures))
        if not report.failures:
            self._reject(round_idx, "no failures on the held-in batch")
            return
        diagnosis = diagnoser.diagnose(self.backend, report.failures)
        self.progress.emit("diagnosis_done", round=round_idx,
                           n_patterns=len(diagnosis),
                           blamed_parts=[d.get("blamed_part", "") for d in diagnosis])

        # 2. Localize: evidence-grounded edit targets + the failing-task evidence.
        round_dir = self.state.candidates_dir / f"round_{round_idx:03d}"
        evidence = {f.task_id: f.trace for f in report.failures if f.trace}
        localization, evidence_dir = self._localize(round_idx, diagnosis, round_dir)

        # 3. Propose several competing whole-strategies (each its own candidate).
        strategies = strategist.propose_many(
            self.backend, self.harness, round_dir, diagnosis,
            n=self.config.loop.strategies_per_round,
            id_prefix=f"r{round_idx}",
            do_not_touch=self.part_map.do_not_touch,
            family_map_text=self._family_map_text(),
            editable_files=self.part_map.editable_files(),
            localization=localization, evidence=evidence, evidence_dir=evidence_dir,
        )
        self.progress.emit("proposal_done", round=round_idx, strategies=[
            {"strategy_id": s.strategy_id, "intent": s.intent} for s in strategies
        ])

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
            self.benchmark, self.split.held_in, self.state.wobble,
            regression_tasks=self.split.regression,  # dual-split when populated (choose_split); [] -> single-split
            borderline_extra_runs=self.config.gate.borderline_extra_runs,
            strict_dual=self.config.gate.strict_dual,
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
            self.progress.emit("gate_decision", round=round_idx,
                               strategy_id=s.strategy_id, additive=additive,
                               **decision_dict(decision))
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
                self.snapshotter.save(self.harness, round_idx, self._held_in_score())
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
        score = self._held_in_score()
        self.state.record(
            RoundOutcome(round_idx, False, 0.0, score, score, note=note)
        )
        self.snapshotter.save(self.harness, round_idx, score)

    # --- mechanism state read by the inner loop each round ---

    def _family_map_text(self) -> str:
        return self.family_map.to_text()

    def _do_not_repeat(self) -> list[str]:
        return self.family_map.do_not_repeat()

    # --- the tree optimizer (treatment arm; classic methods above untouched) ---
    #
    # Per round: diagnose (shared) -> drop non-addressable patterns -> route
    # patterns onto direction nodes -> Thompson-select a direction -> take a
    # pending hypothesis from its frontier (free) or ideate k new ones (one
    # Tier-B call) -> implement ONLY the selected one (one Tier-A call) -> the
    # SAME gate as the classic arm -> the verdict becomes durable tree state:
    # falsified ideas are never re-proposed; noise-killed ideas retry bounded;
    # insights propagate to future ideation.

    def _round_tree(self, round_idx: int) -> None:
        batch = sample_held_in(
            self.split, self.config.piles.round_size, self.config.seed, round_idx
        )
        report = runner.run_batch(self.benchmark, self.harness, batch)
        self.progress.emit("batch_done", round=round_idx,
                           pass_rate=round(report.pass_rate, 4),
                           n_failures=len(report.failures))
        if not report.failures:
            self._reject(round_idx, "no failures on the held-in batch")
            return
        diagnosis = diagnoser.diagnose(self.backend, report.failures)
        self.progress.emit("diagnosis_done", round=round_idx,
                           n_patterns=len(diagnosis),
                           blamed_parts=[d.get("blamed_part", "") for d in diagnosis])
        addressable = [d for d in diagnosis if d.get("addressable", True)]
        if not addressable:
            self._reject(round_idx, "no addressable failure patterns")
            return

        evidence = {f.task_id: f.trace for f in report.failures if f.trace}
        self._route_directions(round_idx, addressable)
        rng = random.Random(self.config.seed * 1_000_003 + round_idx)
        direction = self.tree.select_direction(rng)
        if direction is None:
            self._reject(round_idx, "no selectable directions")
            return
        node = self._select_or_ideate(round_idx, direction, addressable, evidence)
        if node is None:
            self._reject(round_idx, f"ideation produced no hypotheses for {direction.id}")
            return
        self.progress.emit("proposal_done", round=round_idx, strategies=[
            {"strategy_id": node.id, "intent": node.title}
        ])

        round_dir = self.state.candidates_dir / f"round_{round_idx:03d}"
        localization, evidence_dir = self._localize(round_idx, addressable, round_dir)
        strategy = strategist.implement_hypothesis(
            self.backend, self.harness, round_dir / node.id, node, addressable,
            strategy_id=f"r{round_idx}-{node.id}",
            do_not_touch=self.part_map.do_not_touch,
            validated_insights=self.tree.validated_insights(direction.id),
            editable_files=self.part_map.editable_files(),
            localization=localization, evidence=evidence, evidence_dir=evidence_dir,
        )
        survivors = self._shell_filter([strategy])
        if not survivors:
            self._burn_retry(round_idx, node, "implementation dropped at the shell")
            self._reject(round_idx, f"{node.id}: implementation dropped at the shell")
            return
        self._test_tree(round_idx, direction, node, survivors[0], addressable)

    def _route_directions(self, round_idx: int, patterns: list[dict]) -> None:
        """Assign this round's failure patterns to direction nodes (the only
        place directions are created)."""
        assignments = ideator.assign_directions(
            self.backend, self.tree.directions(), patterns
        )
        by_pattern = {p.get("pattern_id"): p for p in patterns}
        for a in assignments:
            if a.get("direction_id"):
                continue  # routed onto an existing direction
            p = by_pattern.get(a.get("pattern_id"), {})
            node = self.tree.add_direction(
                (a.get("new_title") or p.get("root_cause") or a.get("pattern_id") or "?")[:80],
                a.get("new_mechanism") or p.get("agent_mechanism", ""),
                {
                    "verifier_cause": p.get("verifier_cause", ""),
                    "agent_mechanism": p.get("agent_mechanism", ""),
                    "addressable": True,
                },
                round_idx,
            )
            self.progress.emit("tree_mutation", round=round_idx,
                               **mutation_event(node, "created"))

    def _select_or_ideate(self, round_idx: int, direction, diagnosis: list[dict],
                          evidence: dict | None = None):
        """Frontier first: a pending (or retryable noise-killed) hypothesis is
        consumed WITHOUT a new ideation call — paid-for ideas are not
        regenerated. Only an empty frontier buys one Tier-B ideation."""
        frontier = self.tree.frontier(direction.id)
        if frontier:
            node = frontier[0]
            if node.status == "rejected_noise":
                self.tree.mark_noise_retry(node.id)
                self.progress.emit("tree_mutation", round=round_idx,
                                   **mutation_event(node, "noise_retry"))
            return node
        hyps = ideator.ideate(
            self.backend, direction, diagnosis=diagnosis,
            validated_insights=self.tree.validated_insights(direction.id),
            falsified=self.tree.falsified_constraints(),
            pending=self.tree.pending_titles(),
            k=self.config.loop.hypotheses_per_direction,
            trace_evidence=evidence,
        )
        nodes = []
        for h in hyps:
            if not isinstance(h, dict) or not h.get("title"):
                continue
            node = self.tree.add_hypothesis(
                direction.id, title=str(h.get("title", "")),
                mechanism=str(h.get("mechanism", "")),
                hypothesis=str(h.get("hypothesis", "")),
                observable=str(h.get("observable", "")),
                round_idx=round_idx,
            )
            self.progress.emit("tree_mutation", round=round_idx,
                               **mutation_event(node, "created"))
            nodes.append(node)
        return nodes[0] if nodes else None

    def _test_tree(self, round_idx: int, direction, node, s: Strategy,
                   diagnosis: list[dict]) -> None:
        gate = Gate(
            self.benchmark, self.split.held_in, self.state.wobble,
            regression_tasks=self.split.regression,
            borderline_extra_runs=self.config.gate.borderline_extra_runs,
            strict_dual=self.config.gate.strict_dual,
        )
        struct = structural_check.check(
            s.candidate, self.benchmark, backend=self.backend,
            do_not_touch=self.part_map.do_not_touch,
            allow_repair=self.config.edits.allow_repair,
        )
        if not struct.ok:
            self.state.avoid_list.append(struct.error)
            self._burn_retry(round_idx, node, f"structural: {struct.error}")
            self.state.health.gate_rejections += 1
            self._reject(round_idx, f"structural check failed: {struct.error}", empty=False)
            return
        if struct.repaired:
            res = shell.enforce(
                self.harness, s.candidate, self.part_map,
                budget_per_part=self.config.edits.budget_per_part,
            )
            if not res.ok or not res.changed_parts:
                self._burn_retry(round_idx, node, "repair violated shell invariants")
                self.state.health.gate_rejections += 1
                self._reject(round_idx, "repair violated shell invariants", empty=False)
                return
        additive = shell.is_strictly_additive(self.harness, s.candidate)
        decision = gate.evaluate(self.harness, s.candidate, additive=additive)
        self.progress.emit("gate_decision", round=round_idx,
                           strategy_id=s.strategy_id, additive=additive,
                           **decision_dict(decision))
        evidence = {
            "gain_held_in": round(decision.gain, 4),
            "gain_regression": round(decision.regression_gain, 4),
            "runs_used": decision.runs_used, "borderline": decision.borderline,
        }

        if decision.accept:
            self.harness = s.candidate.copy_to(self.state.harness_dir)
            self.state.health.gate_rejections = 0
            self.state.health.empty_rounds = 0
            self.tree.set_status(node.id, "tested_accepted", evidence=evidence,
                                 tested_round=round_idx)
            self._segment_accepted_nodes.append(node.id)
            lesson = insight.distill(self.backend, node, decision, diagnosis)
            if lesson:
                self.tree.set_insight(node.id, lesson)
            self._refresh_direction_summary(direction)
            self.progress.emit("tree_mutation", round=round_idx,
                               **mutation_event(self.tree.node(node.id), "tested_accepted"))
            label = f"{direction.id}:{direction.title[:40]}"
            self.state.record(RoundOutcome(
                round_idx, True, decision.gain, decision.old_score,
                decision.new_score, family_label=label,
                note=f"{s.strategy_id} accepted: {decision.reason}",
            ))
            self.snapshotter.save(self.harness, round_idx, self._held_in_score())
            return

        status = classify_rejection(decision, self.state.wobble)
        self.tree.set_status(node.id, status, evidence=evidence,
                             tested_round=round_idx)
        lesson = insight.distill(self.backend, node, decision, diagnosis)
        if lesson:
            self.tree.set_insight(node.id, lesson)
        if status == "falsified":
            self._refresh_direction_summary(direction)
        self.progress.emit("tree_mutation", round=round_idx,
                           **mutation_event(self.tree.node(node.id), status))
        self.state.health.gate_rejections += 1
        self._reject(round_idx, f"{s.strategy_id} rejected ({status}): {decision.reason}",
                     empty=False)

    def _burn_retry(self, round_idx: int, node, reason: str) -> None:
        """A hypothesis whose implementation never reached the gate burns one
        of its bounded retries (an unimplementable idea must not loop forever)."""
        if node.status == "pending":
            self.tree.set_status(node.id, "rejected_noise", evidence={"reason": reason})
        self.tree.mark_noise_retry(node.id)
        self.progress.emit("tree_mutation", round=round_idx,
                           **mutation_event(self.tree.node(node.id), f"burned: {reason}"))

    def _refresh_direction_summary(self, direction) -> None:
        tested = [n for n in self.tree.children(direction.id)
                  if n.status in ("tested_accepted", "falsified")]
        if not tested:
            return
        summary = insight.summarize_direction(self.backend, direction, tested)
        if summary:
            self.tree.set_insight(direction.id, summary)

    def _audit_and_update_tree(self, round_idx: int):
        """Tree-mode segment close: deep audit + rewind, with audit traps
        falsifying the segment's accepted nodes (the tree's version of the
        family-map trap rule)."""
        verdict = deep_auditor.audit(
            self.benchmark, self.harness, self.split.held_in,
            best_score=self._best_audit_score, wobble=self.state.wobble,
        )
        if verdict.verdict == "worse":
            self.harness = Harness(self._best_dir).copy_to(self.state.harness_dir)
            for nid in dict.fromkeys(self._segment_accepted_nodes):
                old = self.tree.node(nid)
                node = self.tree.set_status(
                    nid, "falsified",
                    evidence={**old.evidence, "audit_trap": True},
                )
                self.progress.emit("tree_mutation", round=round_idx,
                                   **mutation_event(node, "audit_trap"))
                self._refresh_direction_summary(self.tree.node(node.parent_id))
        elif verdict.verdict == "better":
            self._best_audit_score = verdict.score
            self.harness.copy_to(self._best_dir)
            for nid in dict.fromkeys(self._segment_accepted_nodes):
                old = self.tree.node(nid)
                self.tree.set_status(nid, old.status,
                                     evidence={**old.evidence, "audit_confirmed": True})
        plateau = not self._segment_accepted_nodes
        self._segment_accepted_nodes = []
        return verdict, plateau

    def _segment_boundary_tree(self, round_idx: int) -> None:
        verdict, plateau = self._audit_and_update_tree(round_idx)
        self.progress.emit("segment_boundary", round=round_idx,
                           audit_verdict=verdict.verdict,
                           audit_score=round(verdict.score, 4), plateau=plateau)
        # No meta-agent and no family map here: the tree pivots structurally
        # (falsified constraints + posterior decay redirect selection).
        if self._remap:  # codebase changed as edits landed; re-label it
            self.part_map = mapper.map_harness(self.backend, self.harness)

    def _finalize_tree(self, round_idx: int) -> None:
        verdict, plateau = self._audit_and_update_tree(round_idx)
        self.progress.emit("segment_boundary", round=round_idx, final=True,
                           audit_verdict=verdict.verdict,
                           audit_score=round(verdict.score, 4), plateau=plateau)
