"""tb2_ab_compare: the A/B readout renders from real artifact shapes."""

import json

from examples.tb2_ab_compare import arm_metrics, difference_of_differences, render


def _make_arm(ws, *, lifts, accepts, task_runs, tree=False):
    ws.mkdir(parents=True)
    (ws / "lift.json").write_text(json.dumps({
        "cell": "nexau:gemini", "n_tasks": len(lifts), "lift": 0.04, "se": 0.02,
        "detectable_final": 0.143, "per_task_lift": lifts,
    }))
    with (ws / "evidence.jsonl").open("w") as f:
        for i, acc in enumerate(accepts, 1):
            f.write(json.dumps({"ts": 1.0, "round": i, "accepted": acc,
                                "gain": 0.05 if acc else -0.01,
                                "old_score": 0.5, "new_score": 0.55 if acc else 0.5,
                                "family_label": "", "note": "n"}) + "\n")
    with (ws / "progress.jsonl").open("w") as f:
        f.write(json.dumps({"ts": 1.0, "event": "setup_done", "wobble": 0.12,
                            "task_runs": 100}) + "\n")
        for i, acc in enumerate(accepts, 1):
            f.write(json.dumps({"ts": 1.0, "event": "proposal_done", "round": i,
                                "strategies": [{"strategy_id": "s", "intent": "x"}] * (1 if tree else 2)}) + "\n")
            f.write(json.dumps({"ts": 1.0, "event": "gate_decision", "round": i,
                                "accept": acc}) + "\n")
            f.write(json.dumps({"ts": 1.0, "event": "round_end", "round": i,
                                "task_runs": task_runs * i // len(accepts),
                                "cache_hits": 5, "wall_sec": 60.0}) + "\n")
    if tree:
        (ws / "idea_tree.json").write_text(json.dumps({"nodes": [
            {"id": "d1", "parent_id": None, "kind": "direction", "title": "t",
             "status": "pending"},
            {"id": "d1h1", "parent_id": "d1", "kind": "hypothesis", "title": "a",
             "status": "tested_accepted"},
            {"id": "d1h2", "parent_id": "d1", "kind": "hypothesis", "title": "b",
             "status": "falsified"},
        ]}))


def test_compare_end_to_end(tmp_path):
    tasks = [f"t{i}" for i in range(5)]
    _make_arm(tmp_path / "classic", lifts={t: 0.0 for t in tasks},
              accepts=[True, False, True], task_runs=300)
    _make_arm(tmp_path / "tree", lifts={t: (0.34 if t == "t0" else 0.0) for t in tasks},
              accepts=[True, True, False], task_runs=200, tree=True)

    classic = arm_metrics(tmp_path / "classic")
    tree = arm_metrics(tmp_path / "tree")
    assert classic["accepts"] == 2 and tree["accepts"] == 2
    assert classic["tier_a_runs"] == 6 and tree["tier_a_runs"] == 3
    assert tree["tree"]["falsified"] == 1

    dod = difference_of_differences(classic, tree)
    assert dod["n"] == 5
    assert abs(dod["mean"] - 0.068) < 1e-9
    assert "WITHIN NOISE" in dod["verdict"]  # CI spans 0 on 5 tasks

    text = render(classic, tree, dod)
    assert "## Primary (locked test)" in text
    assert "Tier-A / accept" in text
    assert "1 falsified" in text


def test_compare_clear_win(tmp_path):
    tasks = [f"t{i}" for i in range(6)]
    _make_arm(tmp_path / "classic", lifts={t: 0.0 for t in tasks},
              accepts=[True], task_runs=100)
    _make_arm(tmp_path / "tree", lifts={t: 0.3 for t in tasks},
              accepts=[True], task_runs=100, tree=True)
    dod = difference_of_differences(arm_metrics(tmp_path / "classic"),
                                    arm_metrics(tmp_path / "tree"))
    assert "TREE WINS" in dod["verdict"]
