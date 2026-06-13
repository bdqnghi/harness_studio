from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from examples import tb2_self_compare as compare
from studio.backends.mock import MockBackend
from studio.benchmark.base import Benchmark
from studio.components.splitter import SplitPlan, TaskSplit
from studio.harness import Harness
from studio.parts import PartMap, PartType


class _ProtocolBench(Benchmark):
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    def list_tasks(self):
        return []

    def run(self, harness, task_ids, *, run_idx=0):
        tasks = tuple(task_ids)
        self.calls.append(tasks)
        enabled = (harness.root / "extras" / "locked_fix.txt").exists()
        return {
            task: (1.0 if task != "practice" and (task != "locked" or enabled) else 0.0)
            for task in task_ids
        }


def _args(tmp_path):
    return SimpleNamespace(
        harness="nexau",
        backbone="gemini",
        provider=None,
        model=None,
        tasks="practice,judge,regression,audit,locked",
        task_cache=str(tmp_path / "cache"),
        ahe_dir=str(tmp_path / "ahe"),
        workspace=str(tmp_path / "workspace"),
        dry_run=False,
        calibration_k=3,
        opt_k=1,
        test_k=3,
        round_size=1,
        rounds=1,
        segment_length=10,
        strategies=1,
        wobble_runs=3,
        borderline_runs=1,
        budget=4,
        n_concurrent=1,
        timeout_multiplier=1.0,
        seed=7,
        sigma2_prior=0.2,
        delta_round=0.12,
        reg_cap=16,
        test_floor=1,
        test_budget_cap=0,
        heavy_sec=3600.0,
        baseline_score=None,
        baseline_json=None,
        baseline_sigma2=None,
        optimizer="classic",
        hypotheses=4,
        score_cache=None,
        calibrate_only=False,
        baseline_out=None,
    )


def test_backbone_normalization_is_adapter_specific():
    spec = compare.BackboneSpec("openai", "gpt-5.4")
    assert spec.litellm_model == "openai/gpt-5.4"
    assert spec.nexau_model == "gpt-5.4"

    qualified = compare.BackboneSpec("openai", "openai/gpt-5.4")
    assert qualified.litellm_model == "openai/gpt-5.4"
    assert qualified.nexau_model == "gpt-5.4"


def test_provided_baseline_uses_only_frozen_held_in_tasks(tmp_path):
    args = _args(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"held": 0.5, "locked": 1.0}))
    args.baseline_json = str(baseline)

    cal = compare._provided_baseline(args, ["held"], {"held": 10.0})

    assert set(cal.stats) == {"held"}
    assert cal.stats["held"].p == 0.5
    assert cal.sigma2 == 0.25


def test_provided_baseline_rejects_missing_held_in_task(tmp_path):
    args = _args(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"other": 0.5}))
    args.baseline_json = str(baseline)

    with pytest.raises(ValueError, match="missing held-in"):
        compare._provided_baseline(args, ["held"], {})


def test_aggregate_baseline_requires_explicit_noise(tmp_path):
    args = _args(tmp_path)
    args.baseline_score = 0.6

    with pytest.raises(ValueError, match="baseline-sigma2"):
        compare._provided_baseline(args, ["held"], {})


def test_locked_tasks_stay_outside_optimizer_and_live_harness_is_graded(
    tmp_path, monkeypatch,
):
    args = _args(tmp_path)
    src_root = tmp_path / "source"
    src_root.mkdir()
    (src_root / "base.txt").write_text("base\n")
    src = Harness(src_root)

    opt_bench = _ProtocolBench()
    calibration_bench = _ProtocolBench()
    test_bench = _ProtocolBench()
    targets = iter((opt_bench, calibration_bench, test_bench))
    part_map = PartMap(
        parts={PartType.TOOL_CODE: ["extras/"]},
        do_not_touch=["base.txt"],
    )

    def fake_target(*unused_args, **unused_kwargs):
        return next(targets), src, part_map, "openai/gpt-5.4"

    split = TaskSplit(
        practice=["practice"],
        judging=["judge"],
        regression=["regression"],
        audit=["audit"],
        final_exam=["locked"],
    )
    plan = SplitPlan(
        mode="holdout",
        k=3,
        split=split,
        sigma2=0.2,
        n_pool=1,
        n_judging=1,
        n_regression=1,
        n_test=1,
        detectable_round=1.0,
        detectable_final=1.0,
        recommend="split",
    )

    def add_locked_capability(root):
        (root / "extras").mkdir()
        (root / "extras" / "locked_fix.txt").write_text("enabled\n")

    backend = MockBackend(
        json_responses={
            "diagnoser": [[{
                "pattern_id": "p",
                "description": "practice failed",
                "root_cause": "missing capability",
                "failing_task_ids": ["practice"],
                "blamed_part": "tool_code",
                "confidence": 1.0,
            }]],
            "reviewer": [{"keep": [], "drop": []}],
        },
        agent_actions={"strategist": [add_locked_capability]},
    )

    monkeypatch.setattr(compare, "make_target", fake_target)
    monkeypatch.setattr(compare, "make_backend", lambda *a, **kw: backend)
    monkeypatch.setattr(compare, "choose_split", lambda *a, **kw: plan)
    monkeypatch.setattr(
        compare, "read_task_timeouts",
        lambda tasks, cache: {task: 600.0 for task in tasks},
    )
    monkeypatch.setattr(compare, "read_difficulty_meta", lambda tasks, cache: {})

    result = compare.run_cell(args)

    assert all("locked" not in call for call in opt_bench.calls)
    assert all("locked" not in call for call in calibration_bench.calls)
    assert calibration_bench.calls == [("practice", "regression")]
    assert test_bench.calls == [("locked",), ("locked",)]
    assert result["lift"] == 1.0


def test_matrix_child_command_forwards_protocol_options(tmp_path):
    args = _args(tmp_path)
    args.optimizer = "tree"
    args.score_cache = str(tmp_path / "scores.jsonl")
    cmd = compare._matrix_child_cmd(
        args, harness="mini-swe", backbone="gpt-5.4",
        workspace=tmp_path / "cell",
    )
    for flag in (
        "--tasks", "--ahe-dir", "--calibration-k", "--segment-length",
        "--strategies", "--wobble-runs", "--budget", "--seed", "--heavy-sec",
        "--optimizer", "--hypotheses", "--score-cache",
    ):
        assert flag in cmd
    assert cmd[cmd.index("--optimizer") + 1] == "tree"


def test_cell_config_routes_optimizer_and_score_cache(tmp_path):
    args = _args(tmp_path)
    args.optimizer = "tree"
    args.hypotheses = 2
    split = TaskSplit(practice=["p"], judging=["j"], regression=["r"],
                      audit=["a"], final_exam=[])
    cfg = compare._cell_config(split, seed=3, args=args,
                               score_cache=str(tmp_path / "scores.jsonl"))
    assert cfg.loop.optimizer == "tree"
    assert cfg.loop.hypotheses_per_direction == 2
    assert cfg.score_cache == str(tmp_path / "scores.jsonl")
    assert cfg.seed == 3


def test_provided_baseline_accepts_baseline_out_export(tmp_path):
    """The --baseline-out export ({rates, sigma2, ...}) feeds straight back into
    --baseline-json — the calibrate-once-feed-both-arms flow."""
    args = _args(tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "rates": {"held": 0.4, "locked": 1.0},
        "sigma2": 0.18, "model": "gemini/gemini-3.5-flash",
        "calibration_k": 3, "baseline_hash": "abc",
    }))
    args.baseline_json = str(baseline)

    cal = compare._provided_baseline(args, ["held"], {"held": 10.0})
    assert set(cal.stats) == {"held"}
    assert cal.stats["held"].p == 0.4
    assert cal.sigma2 == 0.18  # the exported sigma2 is adopted

    # An explicit --baseline-sigma2 still overrides the exported one.
    args.baseline_sigma2 = 0.22
    cal2 = compare._provided_baseline(args, ["held"], {"held": 10.0})
    assert cal2.sigma2 == 0.22
