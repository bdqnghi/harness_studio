"""Edge-case hardening tests across components."""

from studio.benchmark.toy import ToyBenchmark, build_toy_harness
from studio.benchmark import toy_fixes
from studio.components.family_map import FamilyMap
from studio.components.gate import Gate
from studio.components.splitter import split_tasks
from studio.config import PileConfig


def test_gate_empty_judging_set_does_not_crash(tmp_path):
    bench = ToyBenchmark(per_family=2)
    old = build_toy_harness(tmp_path / "o")
    new = old.copy_to(tmp_path / "n")
    toy_fixes.enable_upper(new.root)
    d = Gate(bench, [], wobble=0.0).evaluate(old, new)
    assert not d.accept and d.gain == 0.0


def test_splitter_caps_when_piles_exceed_tasks():
    tasks = [f"t{i}" for i in range(5)]
    piles = PileConfig(round_size=10, regression=10, held_out=10)
    split = split_tasks(tasks, piles, seed=1)
    allocated = split.held_in + split.regression + split.held_out
    assert sorted(allocated) == sorted(tasks)  # disjoint + total preserved
    assert len(set(allocated)) == len(tasks)   # no duplicates


def test_splitter_piles_are_disjoint():
    tasks = [f"t{i}" for i in range(40)]
    s = split_tasks(tasks, PileConfig(regression=8), seed=7)
    sets = [set(s.held_in), set(s.regression), set(s.held_out)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_family_map_name_parsing_handles_colons_in_reason():
    fm = FamilyMap()
    fm.falsify("middleware", "trap: failed audit: twice")
    assert fm.do_not_repeat() == ["middleware"]


def test_family_map_promote_then_falsify_removes_from_works():
    fm = FamilyMap()
    fm.promote("tool_code", "helped once")
    fm.falsify("tool_code", "later revealed as a trap")
    assert "tool_code" not in fm._family_names(fm.works)
    assert "tool_code" in fm.do_not_repeat()


def test_harness_copy_to_overwrites_existing(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    dest = tmp_path / "d"
    h.copy_to(dest)
    (dest / "extra.txt").write_text("stale")
    h.copy_to(dest)  # overwrite
    assert not (dest / "extra.txt").exists()  # stale file gone


def test_content_hash_ignores_pycache(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    before = h.content_hash()
    (h.root / "__pycache__").mkdir()
    (h.root / "__pycache__" / "x.pyc").write_text("junk")
    assert h.content_hash() == before  # ignored dirs don't affect identity
