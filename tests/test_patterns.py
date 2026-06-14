"""FailurePattern aggregation: noise-aware expected-win ranking, bundling, brief."""

from studio.stages.optimize.diagnose.patterns import (
    ProposalBrief, aggregate, choose_target,
)


def _diag(pid, n, *, addressable=True, conf=1.0, diffs=None):
    return {"pattern_id": pid, "description": f"pat {pid}", "failing_task_ids": [f"{pid}-{i}" for i in range(n)],
            "tasks_affected": n, "addressable": addressable, "confidence": conf,
            "verifier_cause": f"db:{pid}", "blamed_part": "instructions",
            "gt_diff_samples": diffs or [f"expected {pid}"]}


def test_aggregate_ranks_by_expected_win():
    diag = [_diag("a", 2), _diag("b", 6), _diag("c", 3)]
    pats = aggregate(diag, held_in_size=16, noise_floor=0.0)
    assert [p.pattern_id for p in pats] == ["b", "c", "a"]   # by reach (== expected_win here)
    b = pats[0]
    assert b.tasks_affected == 6 and abs(b.max_gain - 6/16) < 1e-9
    assert abs(b.expected_win - 6/16) < 1e-9


def test_non_addressable_sinks_below_addressable():
    pats = aggregate([_diag("big", 8, addressable=False), _diag("small", 2)],
                     held_in_size=16, noise_floor=0.0)
    assert pats[0].pattern_id == "small"       # addressable wins despite smaller reach
    assert pats[1].expected_win == 0.0         # non-addressable -> zero EV


def test_unwinnable_flag_vs_noise_floor():
    # reach 2/16 = 0.125 < 0.25 floor -> unwinnable; 6/16 = 0.375 > floor -> winnable
    pats = aggregate([_diag("a", 2), _diag("b", 6)], held_in_size=16, noise_floor=0.25)
    by = {p.pattern_id: p for p in pats}
    assert by["a"].unwinnable is True
    assert by["b"].unwinnable is False


def test_choose_target_bundles_when_all_unwinnable():
    # each pattern alone is below the floor, but together they clear it
    pats = aggregate([_diag("a", 2), _diag("b", 2), _diag("c", 2)],
                     held_in_size=16, noise_floor=0.25)
    target, bundled = choose_target(pats, held_in_size=16, noise_floor=0.25)
    assert bundled is True
    assert target.pattern_id == "bundle"
    assert target.tasks_affected == 6          # union of all addressable failures
    assert target.unwinnable is False          # 6/16 = 0.375 clears the floor


def test_choose_target_picks_top_when_winnable():
    pats = aggregate([_diag("a", 2), _diag("b", 6)], held_in_size=16, noise_floor=0.1)
    target, bundled = choose_target(pats, held_in_size=16, noise_floor=0.1)
    assert bundled is False and target.pattern_id == "b"


def test_proposal_brief_render_is_class_level_and_grounded():
    pats = aggregate([_diag("b", 6, diffs=["expected modify_order(qty=1)"])],
                     held_in_size=16, noise_floor=0.0)
    brief = ProposalBrief.from_pattern(pats[0], held_in_size=16).render()
    assert "6 of 16" in brief
    assert "expected modify_order(qty=1)" in brief          # ground-truth diff surfaced
    assert "GENERAL" in brief and "never a task-specific" in brief   # class-level guard
