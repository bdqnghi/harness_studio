"""IdeaTree core: persistence, statuses, posteriors, selection, constraints.

The tree is the treatment arm's memory — every property here is load-bearing
for the A/B experiment (falsified ideas must never be re-bought; noise-killed
ideas must stay re-proposable; the whole thing must survive a crash)."""

import json
import random

import pytest

from studio.stages.optimize.gate import GateDecision
from studio.stages.optimize.idea_tree import (
    MAX_NOISE_RETRIES, IdeaTree, classify_rejection, mutation_event,
)


def _tree(tmp_path, **kw):
    return IdeaTree.load_or_create(tmp_path / "idea_tree.json", **kw)


def _seed(tree, round_idx=1):
    d = tree.add_direction("context overflow", "agent loses task state",
                           {"verifier_cause": "timeout", "agent_mechanism": "loops",
                            "addressable": True}, round_idx)
    h = tree.add_hypothesis(d.id, title="truncate observations",
                            mechanism="cap tool output", hypothesis="cap at 2k chars",
                            observable="long-output tasks stop timing out",
                            round_idx=round_idx)
    return d, h


def test_persistence_round_trip(tmp_path):
    tree = _tree(tmp_path)
    d, h = _seed(tree)
    tree.set_status(h.id, "tested_accepted", evidence={"gain_judging": 0.05},
                    tested_round=2)
    tree.set_insight(h.id, "capping output fixed two timeout tasks")

    reloaded = _tree(tmp_path)
    assert set(reloaded.nodes) == {d.id, h.id}
    rh = reloaded.node(h.id)
    assert rh.status == "tested_accepted"
    assert rh.evidence == {"gain_judging": 0.05}
    assert rh.tested_round == 2
    assert "capping output" in rh.insight


def test_save_is_atomic_no_tmp_left(tmp_path):
    tree = _tree(tmp_path)
    _seed(tree)
    assert not list(tmp_path.glob("*.tmp"))
    # The file on disk is always complete JSON.
    data = json.loads((tmp_path / "idea_tree.json").read_text())
    assert len(data["nodes"]) == 2


def test_ids_and_children(tmp_path):
    tree = _tree(tmp_path)
    d1, h1 = _seed(tree)
    h2 = tree.add_hypothesis(d1.id, title="t2", mechanism="m", hypothesis="h",
                             observable="o", round_idx=1)
    d2 = tree.add_direction("second", "m", {}, 2)
    assert (d1.id, d2.id) == ("d1", "d2")
    assert (h1.id, h2.id) == ("d1h1", "d1h2")
    assert [n.id for n in tree.children(d1.id)] == ["d1h1", "d1h2"]
    with pytest.raises(ValueError):
        tree.add_hypothesis(h1.id, title="x", mechanism="m", hypothesis="h",
                            observable="o", round_idx=1)  # parent must be a direction


def test_posterior_counts(tmp_path):
    tree = _tree(tmp_path)
    d, h1 = _seed(tree)
    h2 = tree.add_hypothesis(d.id, title="t2", mechanism="m", hypothesis="h2",
                             observable="o", round_idx=1)
    h3 = tree.add_hypothesis(d.id, title="t3", mechanism="m", hypothesis="h3",
                             observable="o", round_idx=1)
    tree.set_status(h1.id, "tested_accepted")
    tree.set_status(h2.id, "falsified")
    tree.set_status(h3.id, "rejected_noise")
    alpha, beta = tree.posterior(d.id)
    assert alpha == 2.0          # 1 + 1 accepted
    assert beta == 2.5           # 1 + 1 falsified + 0.5 noise


def test_frontier_order_and_noise_cutoff(tmp_path):
    tree = _tree(tmp_path)
    d, h1 = _seed(tree)
    h2 = tree.add_hypothesis(d.id, title="t2", mechanism="m", hypothesis="h2",
                             observable="o", round_idx=1)
    tree.set_status(h1.id, "rejected_noise")
    # Pending first, then the retryable noise rejection.
    assert [n.id for n in tree.frontier(d.id)] == [h2.id, h1.id]
    for _ in range(MAX_NOISE_RETRIES):
        tree.mark_noise_retry(h1.id)
    assert [n.id for n in tree.frontier(d.id)] == [h2.id]  # retries exhausted
    tree.set_status(h2.id, "falsified")
    assert tree.frontier(d.id) == []  # falsified never reappears


def test_constraints_and_insights(tmp_path):
    tree = _tree(tmp_path)
    d, h1 = _seed(tree)
    tree.set_status(h1.id, "falsified")
    tree.set_insight(h1.id, "made timeouts worse")
    tree.set_insight(d.id, "the real culprit is observation size")
    h2 = tree.add_hypothesis(d.id, title="t2", mechanism="m", hypothesis="h2",
                             observable="o", round_idx=2)
    tree.set_status(h2.id, "tested_accepted")
    tree.set_insight(h2.id, "worked on retry tasks")

    cons = tree.falsified_constraints()
    assert len(cons) == 1
    assert "truncate observations" in cons[0] and "made timeouts worse" in cons[0]
    insights = tree.validated_insights(d.id)
    assert "the real culprit is observation size" in insights
    assert "worked on retry tasks" in insights
    assert tree.pending_titles() == []


def test_insight_truncated_to_200_words(tmp_path):
    tree = _tree(tmp_path)
    d, h = _seed(tree)
    tree.set_insight(h.id, " ".join(["word"] * 500))
    assert len(tree.node(h.id).insight.split()) == 200


def test_select_direction_deterministic_and_evidence_driven(tmp_path):
    tree = _tree(tmp_path)
    d1, _ = _seed(tree)
    d2 = tree.add_direction("second", "m", {}, 1)
    # Determinism: identical seeds pick identically.
    picks_a = [tree.select_direction(random.Random(s)).id for s in range(20)]
    picks_b = [tree.select_direction(random.Random(s)).id for s in range(20)]
    assert picks_a == picks_b
    # At n=0 both directions get explored.
    assert set(picks_a) == {d1.id, d2.id}
    # Pile falsifications onto d1 -> selection shifts strongly to d2.
    for i in range(6):
        h = tree.add_hypothesis(d1.id, title=f"f{i}", mechanism="m",
                                hypothesis="h", observable="o", round_idx=1)
        tree.set_status(h.id, "falsified")
    for i in range(2):
        h = tree.add_hypothesis(d2.id, title=f"w{i}", mechanism="m",
                                hypothesis="h", observable="o", round_idx=1)
        tree.set_status(h.id, "tested_accepted")
    picks = [tree.select_direction(random.Random(s)).id for s in range(50)]
    assert picks.count(d2.id) > 40
    # A falsified direction is never selected.
    tree.set_status(d2.id, "falsified")
    picks = {tree.select_direction(random.Random(s)).id for s in range(20)}
    assert picks == {d1.id}


def test_select_direction_empty_tree(tmp_path):
    tree = _tree(tmp_path)
    assert tree.select_direction(random.Random(0)) is None


def test_classify_rejection_all_gate_shapes():
    wobble = 0.10
    clear = GateDecision(False, -0.20, 0.5, 0.3, regressed=True, runs_used=1)
    assert classify_rejection(clear, wobble) == "falsified"
    # Borderline-resolved tiny negative: regressed=True but inside residual noise.
    borderline = GateDecision(False, -0.01, 0.5, 0.49, regressed=True,
                              borderline=True, runs_used=6)
    assert classify_rejection(borderline, wobble) == "rejected_noise"
    # Borderline-resolved but clearly negative even after averaging.
    resolved_bad = GateDecision(False, -0.30, 0.5, 0.2, regressed=True,
                                borderline=True, runs_used=6)
    assert classify_rejection(resolved_bad, wobble) == "falsified"
    # Dual-split: judging fine, regression split clearly regressed.
    dual = GateDecision(False, 0.02, 0.5, 0.52, regressed=True, runs_used=1,
                        regression_gain=-0.25)
    assert classify_rejection(dual, wobble) == "falsified"
    # Not regressed at all (neutral behavioral edit) -> noise.
    neutral = GateDecision(False, 0.0, 0.5, 0.5, regressed=False, runs_used=1)
    assert classify_rejection(neutral, wobble) == "rejected_noise"


def test_markdown_render(tmp_path):
    md_path = tmp_path / "tree.md"
    tree = _tree(tmp_path, md_path=md_path)
    d, h = _seed(tree)
    tree.set_status(h.id, "falsified")
    text = md_path.read_text()
    assert "# Hypothesis tree" in text
    assert "d1 context overflow" in text
    assert "[x] d1h1" in text


def test_mutation_event_payload(tmp_path):
    tree = _tree(tmp_path)
    d, h = _seed(tree)
    ev = mutation_event(h, "created")
    assert ev == {"node": "d1h1", "kind": "hypothesis",
                  "title": "truncate observations", "status": "pending",
                  "change": "created"}
