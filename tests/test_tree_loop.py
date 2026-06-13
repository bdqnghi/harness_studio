"""Integration: the tree optimizer climbs the toy via the same gate as classic.

Round 1 ideates two hypotheses and tests a known-bad one (falsified). Round 2
proves frontier reuse: the pending sibling is consumed with NO second ideation
call (the absence of a scripted ideator response IS the assertion). Round 3
ideates again and must carry the falsified constraint + validated insights in
its prompt. Statuses, insights, persistence, audit traps, and the A/B flag
routing are all asserted against the same run.
"""

import json

from studio.backends.mock import MockBackend
from studio.benchmark import toy_fixes
from studio.benchmark.toy import FAMILIES, ToyBenchmark, build_toy_harness, toy_part_map
from studio.components.evidence import (
    EvidenceStore, TaskEvidence, TraceWindow, VerifierSignal, to_flat_excerpt,
)
from studio.components.ideator import CONSTRAINT_HEADER
from studio.components.splitter import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.orchestrator import Orchestrator


class EvidenceToyBenchmark(ToyBenchmark):
    """ToyBenchmark that records structured evidence for failing tasks, so the
    localizer + editor have a corpus to work from (the real adapters do this)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.evidence_store = EvidenceStore()

    def run(self, harness, task_ids, *, run_idx=0):
        scores = super().run(harness, task_ids, run_idx=run_idx)
        h = harness.content_hash()
        for tid, s in scores.items():
            if s < 1.0:
                self.evidence_store.put(h, TaskEvidence(
                    task_id=tid, reward=s,
                    signals=[VerifierSignal("test", tid.split("-")[0], False, "op failed")],
                    windows=[TraceWindow(tid, 0, 0, 0,
                             [{"role": "assistant", "content": f"FAILMARK-{tid}"}], "failed")],
                ))
        return scores

    def last_trace(self, task_id, *, harness=None):
        if harness is None:
            return ""
        ev = self.evidence_store.get(harness.content_hash(), task_id)
        return to_flat_excerpt(ev) if ev else ""

    def last_evidence(self, task_id, *, harness=None):
        if harness is None:
            return None
        return self.evidence_store.get(harness.content_hash(), task_id)


# A localization target that survives the citation guard: pure addition (empty
# current_text) into a real editable file, citing a task that fails every round.
_LOC = {"targets": [{
    "pattern_id": "p1", "target_file": "instructions.txt", "current_text": "",
    "target_locator": "the rules section", "change_kind": "add_rule",
    "evidence": [{"task_id": "add-6", "quote": "FAILMARK-add-6"}],
}]}

SPLIT = TaskSplit(
    held_in=[f"{f}-{i}" for f in FAMILIES for i in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11)],
    held_out=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
)

DIAG = [{
    "pattern_id": "p1", "description": "several ops fail",
    "root_cause": "buggy or disabled ops", "failing_task_ids": ["reverse-6"],
    "blamed_part": "tool_code", "confidence": 0.8,
    "verifier_cause": "wrong output", "agent_mechanism": "op missing or buggy",
    "addressable": True,
}]

NEW_DIRECTION = {"assignments": [{
    "pattern_id": "p1", "direction_id": "",
    "new_title": "broken ops", "new_mechanism": "ops buggy or disabled",
}]}
ROUTE_TO_D1 = {"assignments": [{"pattern_id": "p1", "direction_id": "d1"}]}

HYPS_R1 = {"hypotheses": [
    {"title": "disable echo", "mechanism": "remove distraction",
     "hypothesis": "remove ENABLE echo from instructions",
     "observable": "echo tasks change behavior"},
    {"title": "enable upper", "mechanism": "op exists but is not enabled",
     "hypothesis": "append ENABLE upper to instructions",
     "observable": "upper tasks pass"},
]}
HYPS_R3 = {"hypotheses": [
    {"title": "fix reverse", "mechanism": "buggy implementation",
     "hypothesis": "make _reverse return arg[::-1]",
     "observable": "reverse tasks pass"},
    {"title": "spare idea", "mechanism": "m", "hypothesis": "h", "observable": "o"},
]}


def _tree_backend():
    return MockBackend(
        json_responses={
            "diagnoser": [DIAG] * 3,
            "direction-router": [NEW_DIRECTION, ROUTE_TO_D1, ROUTE_TO_D1],
            # Round 2 has NO ideator response: the frontier must serve it.
            "ideator": [HYPS_R1, HYPS_R3],
            "insight": [
                {"insight": "disabling echo regressed working tasks"},
                {"insight": "enabling disabled ops works"},
                {"insight": "fixing buggy op implementations works"},
            ],
            "insight-direction": [
                {"insight": "direction summary v1"},
                {"insight": "direction summary v2"},
                {"insight": "direction summary v3"},
            ],
        },
        agent_actions={"strategist": [
            toy_fixes.regress_echo,   # round 1: falsified
            toy_fixes.enable_upper,   # round 2: accepted (frontier reuse)
            toy_fixes.fix_reverse,    # round 3: accepted
        ]},
    )


def _config(**loop_kw):
    defaults = dict(rounds=3, wobble_runs=3, 
                    hypotheses_per_direction=2)
    defaults.update(loop_kw)
    return Config(loop=LoopConfig(**defaults), edits=EditConfig(allow_repair=False))


def _run_tree(tmp_path, backend=None, benchmark=None, **loop_kw):
    orch = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src"),
        benchmark=benchmark or ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=backend or _tree_backend(),
        config=_config(**loop_kw),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    return orch.run(), orch


def test_tree_loop_climbs_and_records_statuses(tmp_path):
    result, orch = _run_tree(tmp_path)
    assert result.baseline_final == 0.25
    assert result.final_score == 0.75          # upper + reverse fixed; add still missing
    assert result.accepted == 2                # rounds 2 and 3

    statuses = {n.id: n.status for n in orch.tree.nodes.values()}
    assert statuses == {
        "d1": "pending",
        "d1h1": "falsified",                   # regress_echo: clear regression
        "d1h2": "tested_accepted",             # enable_upper via frontier
        "d1h3": "tested_accepted",             # fix_reverse
        "d1h4": "pending",                     # spare idea stays on the frontier
    }
    # Insights landed on tested nodes and the direction.
    assert orch.tree.node("d1h1").insight == "disabling echo regressed working tasks"
    assert orch.tree.node("d1").insight.startswith("direction summary")
    # The finalize audit confirmed the segment's accepted nodes.
    assert orch.tree.node("d1h2").evidence.get("audit_confirmed") is True
    assert orch.tree.node("d1h3").evidence.get("audit_confirmed") is True


def test_two_stage_order_and_frontier_reuse(tmp_path):
    _, orch = _run_tree(tmp_path)
    backend = orch.backend
    # Exactly 2 ideation calls for 3 rounds: round 2 consumed the frontier.
    assert sum(1 for k, t in backend.calls if t == "ideator") == 2
    assert sum(1 for k, t in backend.calls if t == "strategist") == 3
    # Stage 1 (text) precedes stage 2 (implementation) in round 1.
    kinds = [(k, t) for k, t in backend.calls if t in ("ideator", "strategist")]
    assert kinds[0] == ("prompt_json", "ideator")
    assert kinds[1] == ("run_agent", "strategist")
    # Tree mode never consults the classic-only helpers.
    tags = {t for _, t in backend.calls}
    assert not tags & {"reviewer", "ranker", "meta"}


def test_constraints_and_insights_reach_ideation(tmp_path):
    _, orch = _run_tree(tmp_path)
    ideator_prompts = [p for t, p in orch.backend.prompt_log if t == "ideator"]
    assert len(ideator_prompts) == 2
    round3 = ideator_prompts[1]
    assert CONSTRAINT_HEADER in round3
    assert "remove ENABLE echo from instructions" in round3   # falsified hypothesis text
    assert "enabling disabled ops works" in round3            # validated sibling insight
    # The implementer received the hypothesis verbatim as a fixed contract.
    impl = [p for t, p in orch.backend.prompt_log if t == "strategist"]
    assert "Implement EXACTLY this hypothesis" in impl[0]
    assert "remove ENABLE echo from instructions" in impl[0]


def test_tree_persists_and_resumes(tmp_path):
    _, orch = _run_tree(tmp_path)
    tree_file = orch.state.root / "idea_tree.json"
    assert tree_file.exists()
    assert (orch.state.root / "tree.md").exists()
    # A fresh orchestrator over the same workspace reloads the full tree.
    orch2 = Orchestrator(
        workspace=tmp_path / "ws",
        source_harness=build_toy_harness(tmp_path / "src2"),
        benchmark=ToyBenchmark(per_family=12, noise_per_mille=0),
        backend=MockBackend(),
        config=_config(),
        split=SPLIT,
        part_map=toy_part_map(),
    )
    assert {n.id for n in orch2.tree.nodes.values()} == {"d1", "d1h1", "d1h2", "d1h3", "d1h4"}
    assert orch2.tree.node("d1h1").status == "falsified"


def test_tree_emits_mutations_and_persists_tree(tmp_path):
    _, orch = _run_tree(tmp_path)
    events = [json.loads(line) for line in orch.state.progress_path.read_text().splitlines()]
    assert any(e["event"] == "tree_mutation" for e in events)
    assert (orch.state.root / "idea_tree.json").exists()  # durable tree state


def test_non_addressable_patterns_stop_the_round(tmp_path):
    backend = MockBackend(json_responses={
        "diagnoser": [[{**DIAG[0], "addressable": False}]],
    })
    result, orch = _run_tree(tmp_path, backend=backend, rounds=1)
    assert result.accepted == 0
    assert "no addressable failure patterns" in result.rounds[0].note
    assert not any(t == "direction-router" for _, t in backend.calls)


# --- Phase 6: localization wired into both paths (applied everywhere) ---

def test_localization_reaches_editor_tree(tmp_path):
    """With localizer on, the tree editor receives the failing-task evidence
    AND a validated localized target — inline mode (single Tier-B call)."""
    bench = EvidenceToyBenchmark(per_family=12, noise_per_mille=0)
    backend = _tree_backend()
    backend._json["localizer"] = [_LOC, _LOC, _LOC]   # one per round
    _, orch = _run_tree(tmp_path, backend=backend, benchmark=bench, localizer="inline")

    events = [json.loads(line) for line in orch.state.progress_path.read_text().splitlines()]
    assert any(e["event"] == "localization_done" for e in events)
    assert ("prompt_json", "localizer") in backend.calls          # inline, not agentic
    assert ("run_explore", "localizer") not in backend.calls
    impl_prompts = [p for t, p in backend.prompt_log if t == "strategist"]
    assert any("FAILMARK-add-6" in p for p in impl_prompts)        # evidence reached editor
    assert any("Localized edit target" in p and "instructions.txt" in p
               for p in impl_prompts)                              # validated target reached editor
    # ideation was also grounded in the transcripts
    assert any("FAILMARK-" in p for t, p in backend.prompt_log if t == "ideator")


def test_localizer_off_makes_no_localizer_calls(tmp_path):
    """Default 'off' -> zero localizer calls, no localization_done events."""
    bench = EvidenceToyBenchmark(per_family=12, noise_per_mille=0)
    _, orch = _run_tree(tmp_path, benchmark=bench)   # localizer defaults to "off"
    tags = {t for _, t in orch.backend.calls}
    assert "localizer" not in tags
    events = [json.loads(line) for line in orch.state.progress_path.read_text().splitlines()]
    assert not any(e["event"] == "localization_done" for e in events)
