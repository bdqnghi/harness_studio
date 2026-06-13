"""Tree-arm AI helpers: router, ideator, stage-2 implementer, insight distiller."""

from studio import schemas
from studio.backends.mock import MockBackend
from studio.stages.optimize import ideator, insight, strategist
from studio.stages.optimize.gate import GateDecision
from studio.stages.optimize.idea_tree import IdeaTree


def _tree(tmp_path):
    return IdeaTree.load_or_create(tmp_path / "idea_tree.json")


PATTERNS = [
    {"pattern_id": "p1", "root_cause": "agent loops on long output",
     "verifier_cause": "timeout", "agent_mechanism": "re-reads giant file",
     "blamed_part": "memory", "failing_task_ids": ["t1"], "addressable": True},
    {"pattern_id": "p2", "root_cause": "missing edit tool",
     "verifier_cause": "file unchanged", "agent_mechanism": "echo-overwrites files",
     "blamed_part": "tool_code", "failing_task_ids": ["t2"], "addressable": True},
]


def test_assign_directions_reuse_and_create(tmp_path):
    tree = _tree(tmp_path)
    d = tree.add_direction("output handling", "agent drowns in tool output", {}, 1)
    backend = MockBackend(json_responses={"direction-router": [{
        "assignments": [
            {"pattern_id": "p1", "direction_id": d.id},
            {"pattern_id": "p2", "direction_id": "",
             "new_title": "editing tools", "new_mechanism": "no surgical edit tool"},
        ],
    }]})
    out = ideator.assign_directions(backend, tree.directions(), PATTERNS)
    assert {a["pattern_id"]: a["direction_id"] for a in out} == {"p1": d.id, "p2": ""}
    # The router saw the existing direction.
    tag, prompt = backend.prompt_log[0]
    assert tag == "direction-router" and "output handling" in prompt


def test_assign_directions_fallback_routes_unmatched(tmp_path):
    backend = MockBackend(json_responses={"direction-router": [{
        "assignments": [{"pattern_id": "p1", "direction_id": ""}],
    }]})
    out = ideator.assign_directions(backend, [], PATTERNS)
    routed = {a["pattern_id"] for a in out}
    assert routed == {"p1", "p2"}  # p2 was unrouted -> deterministic new direction
    p2 = next(a for a in out if a["pattern_id"] == "p2")
    assert p2["direction_id"] == "" and p2["new_title"].startswith("missing edit tool")


def test_assign_directions_empty_patterns_no_call():
    backend = MockBackend()
    assert ideator.assign_directions(backend, [], []) == []
    assert backend.calls == []


HYPS = {"hypotheses": [
    {"title": "truncate tool output", "mechanism": "cap observation size",
     "hypothesis": "cap at 2000 chars in middleware", "observable": "timeouts stop"},
    {"title": "add str_replace tool", "mechanism": "surgical edits",
     "hypothesis": "new tool + registration", "observable": "file-edit tasks pass",
     "conflicts": "none known"},
]}


def test_ideate_prompt_contains_constraints_insights_and_frontier(tmp_path):
    tree = _tree(tmp_path)
    d = tree.add_direction("output handling", "agent drowns in output", {}, 1)
    backend = MockBackend(json_responses={"ideator": [HYPS]})
    out = ideator.ideate(
        backend, d, diagnosis=PATTERNS,
        validated_insights=["capping helped on retry tasks"],
        falsified=["raise step limit: just raise limits — made cost explode"],
        pending=["stream observations"],
        k=2,
    )
    assert len(out) == 2 and out[0]["title"] == "truncate tool output"
    tag, prompt = backend.prompt_log[0]
    assert tag == "ideator"
    assert ideator.CONSTRAINT_HEADER in prompt
    assert "raise step limit" in prompt                 # the falsified ledger text
    assert "capping helped on retry tasks" in prompt    # validated insights
    assert "stream observations" in prompt              # pending frontier
    assert "Propose exactly 2" in prompt


def test_ideate_caps_at_k(tmp_path):
    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    over = {"hypotheses": HYPS["hypotheses"] * 3}  # 6 returned
    backend = MockBackend(json_responses={"ideator": [over]})
    out = ideator.ideate(backend, d, diagnosis=[], validated_insights=[],
                         falsified=[], pending=[], k=4)
    assert len(out) == 4


def test_implement_hypothesis_quotes_contract_verbatim(tmp_path):
    from studio.benchmark.toy import build_toy_harness

    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    node = tree.add_hypothesis(d.id, title="truncate tool output",
                               mechanism="cap observation size",
                               hypothesis="cap at 2000 chars in middleware",
                               observable="timeouts stop", round_idx=1)
    base = build_toy_harness(tmp_path / "src")

    def edit(workspace):
        (workspace / "added.txt").write_text("x")

    backend = MockBackend(agent_actions={"strategist": [edit]})
    strategy = strategist.implement_hypothesis(
        backend, base, tmp_path / "cand", node, PATTERNS,
        strategy_id="r1t1", do_not_touch=["pyproject.toml"],
        validated_insights=["prior lesson"],
    )
    tag, instruction = backend.prompt_log[0]
    assert tag == "strategist"
    assert "Implement EXACTLY this hypothesis" in instruction
    assert "cap at 2000 chars in middleware" in instruction  # verbatim contract
    assert "timeouts stop" in instruction                    # the observable
    assert "prior lesson" in instruction
    assert "pyproject.toml" in instruction
    assert strategy.strategy_id == "r1t1" and strategy.intent == "truncate tool output"
    assert (tmp_path / "cand" / "added.txt").exists()        # edited the copy
    assert not (tmp_path / "src" / "added.txt").exists()     # not the base


def test_implement_hypothesis_states_editable_whitelist(tmp_path):
    """The proposer must be told its editable surface (the bug that dropped
    every tau2 edit: only do_not_touch was passed, so it created new files
    that the shell reverted, leaving the prose policy unchanged)."""
    from studio.benchmark.toy import build_toy_harness

    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    node = tree.add_hypothesis(d.id, title="add refund rule", mechanism="policy",
                               hypothesis="add a refund-eligibility rule",
                               observable="refund tasks pass", round_idx=1)
    base = build_toy_harness(tmp_path / "src")
    backend = MockBackend(agent_actions={"strategist": [lambda ws: None]})
    strategist.implement_hypothesis(
        backend, base, tmp_path / "cand", node, PATTERNS, strategy_id="r1t1",
        editable_files=["policy.md"],
    )
    _, instr = backend.prompt_log[0]
    assert "EDITABLE FILES (you may ONLY change these): policy.md" in instr
    assert "Do NOT create new files" in instr        # single plain file -> no new files
    assert "express the idea as added/rewritten policy rules" in instr.lower() \
        or "prose policy" in instr.lower()

    # A directory entry permits adding files (mini-swe style).
    backend2 = MockBackend(agent_actions={"strategist": [lambda ws: None]})
    strategist.implement_hypothesis(
        backend2, base, tmp_path / "cand2", node, PATTERNS, strategy_id="r1t2",
        editable_files=["tools/", "config.yaml"],
    )
    _, instr2 = backend2.prompt_log[0]
    assert "tools/, config.yaml" in instr2
    assert "add files only under the directory entries" in instr2


def test_implement_hypothesis_carries_localization_and_evidence(tmp_path):
    """Phase 5: the editor SEES the localized span + the cited transcript
    evidence (the read-before-act payload), not just the diagnosis summary."""
    from studio.benchmark.toy import build_toy_harness

    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    node = tree.add_hypothesis(d.id, title="add refund rule", mechanism="policy",
                               hypothesis="add a refund-eligibility rule",
                               observable="refund tasks pass", round_idx=1)
    base = build_toy_harness(tmp_path / "src")
    backend = MockBackend(agent_actions={"strategist": [lambda ws: None]})
    localization = [{
        "target_file": "policy.md", "target_locator": "the conduct section",
        "change_kind": "add_rule", "current_text": "Be helpful.",
        "rationale": "no refund rule exists",
        "evidence": [{"task_id": "t9", "quote": "agent refused a valid refund"}],
    }]
    evidence = {"t9": "reward=0.0\nfailed checks: action process_refund\n[assistant] sorry, no refunds"}
    strategist.implement_hypothesis(
        backend, base, tmp_path / "cand", node, PATTERNS, strategy_id="r1t1",
        editable_files=["policy.md"], localization=localization, evidence=evidence,
    )
    _, instr = backend.prompt_log[0]
    assert "Localized edit target" in instr
    assert "policy.md @ the conduct section" in instr
    assert "agent refused a valid refund" in instr        # cited transcript quote
    assert "Failure evidence" in instr and "process_refund" in instr  # evidence window
    assert "READ THIS before editing" in instr


def test_implement_hypothesis_without_localization_is_unchanged(tmp_path):
    """No localization/evidence -> the appended blocks are absent (byte-identical
    to the pre-Phase-5 prompt). Locks the degrade-to-nothing invariant."""
    from studio.benchmark.toy import build_toy_harness

    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    node = tree.add_hypothesis(d.id, title="t", mechanism="m", hypothesis="h",
                               observable="o", round_idx=1)
    base = build_toy_harness(tmp_path / "src")
    backend = MockBackend(agent_actions={"strategist": [lambda ws: None]})
    strategist.implement_hypothesis(
        backend, base, tmp_path / "cand", node, PATTERNS, strategy_id="r1t1",
        editable_files=["policy.md"],
    )
    _, instr = backend.prompt_log[0]
    assert "Localized edit target" not in instr
    assert "Failure evidence (verifier output" not in instr


def test_ideate_includes_trace_evidence(tmp_path):
    d = _tree(tmp_path).add_direction("dir", "mech", {}, 1)
    backend = MockBackend(json_responses={"ideator": [HYPS]})
    ideator.ideate(backend, d, diagnosis=PATTERNS, validated_insights=[],
                   falsified=[], pending=[], k=2,
                   trace_evidence={"t9": "agent never verified identity"})
    _, prompt = backend.prompt_log[0]
    assert "agent never verified identity" in prompt


def test_insight_distill_and_direction_summary(tmp_path):
    tree = _tree(tmp_path)
    d = tree.add_direction("dir", "m", {}, 1)
    node = tree.add_hypothesis(d.id, title="t", mechanism="m", hypothesis="h",
                               observable="o", round_idx=1)
    decision = GateDecision(True, 0.06, 0.5, 0.56, runs_used=1, reason="clearly better")
    backend = MockBackend(json_responses={
        "insight": [{"insight": "capping output removed the timeout failure mode"}],
        "insight-direction": [{"insight": "output size is the bottleneck"}],
    })
    text = insight.distill(backend, node, decision, PATTERNS)
    assert "timeout failure mode" in text
    summary = insight.summarize_direction(backend, d, [node])
    assert "bottleneck" in summary
    # Distillation failure degrades to "" (never a failed round).
    broken = MockBackend()  # no scripted responses -> helper raises internally
    assert insight.distill(broken, node, decision, []) == ""
    assert insight.summarize_direction(broken, d, [node]) == ""


def test_new_schemas_validate():
    schemas.validate({"assignments": [{"pattern_id": "p", "direction_id": "d1"}]},
                     schemas.DIRECTION_ASSIGN)
    schemas.validate(HYPS, schemas.HYPOTHESES)
    schemas.validate({"insight": "x"}, schemas.INSIGHT)
    for bad, schema in [
        ({"assignments": [{"pattern_id": "p"}]}, schemas.DIRECTION_ASSIGN),
        ({"hypotheses": [{"title": "t"}]}, schemas.HYPOTHESES),
        ({}, schemas.INSIGHT),
    ]:
        try:
            schemas.validate(bad, schema)
            raise AssertionError(f"{bad} should not validate")
        except schemas.SchemaError:
            pass
