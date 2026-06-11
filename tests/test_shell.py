from studio.benchmark.toy import build_toy_harness, toy_part_map
from studio.benchmark import toy_fixes
from studio.components import shell
from studio.parts import PartType


def _pair(tmp_path):
    original = build_toy_harness(tmp_path / "orig")
    candidate = original.copy_to(tmp_path / "cand")
    return original, candidate


def test_allows_in_budget_edit(tmp_path):
    original, candidate = _pair(tmp_path)
    toy_fixes.fix_reverse(candidate.root)
    res = shell.enforce(original, candidate, toy_part_map(), budget_per_part=3)
    assert res.ok
    assert res.changed_parts[PartType.TOOL_CODE] == ["tools.py"]
    assert res.reverted == []


def test_reverts_do_not_touch_edit(tmp_path):
    original, candidate = _pair(tmp_path)
    pmap = toy_part_map()
    pmap.do_not_touch = ["config.json"]
    pmap.parts[PartType.MIDDLEWARE] = []  # config.json no longer editable
    # touch the now do-not-touch file
    (candidate.root / "config.json").write_text('{"hacked": true}\n')
    res = shell.enforce(original, candidate, pmap, budget_per_part=3)
    assert "config.json" in res.reverted
    assert candidate.read_file("config.json") == original.read_file("config.json")


def test_deletes_newly_created_unmapped_file(tmp_path):
    original, candidate = _pair(tmp_path)
    (candidate.root / "sneaky.py").write_text("print('hi')\n")
    res = shell.enforce(original, candidate, toy_part_map(), budget_per_part=3)
    assert "sneaky.py" in res.reverted
    assert not (candidate.root / "sneaky.py").exists()


def test_budget_overflow_rejects(tmp_path):
    original, candidate = _pair(tmp_path)
    # two changes to the same part with a budget of 1
    toy_fixes.fix_reverse(candidate.root)
    pmap = toy_part_map()
    pmap.parts[PartType.TOOL_CODE] = ["tools.py", "config.json"]
    pmap.parts[PartType.MIDDLEWARE] = []
    (candidate.root / "config.json").write_text('{"x": 1}\n')
    res = shell.enforce(original, candidate, pmap, budget_per_part=1)
    assert not res.ok
    assert res.violations and "tool_code" in res.violations[0]


def test_strict_additive_requires_only_new_files(tmp_path):
    original, candidate = _pair(tmp_path)
    assert not shell.is_strictly_additive(original, candidate)

    pmap = toy_part_map()
    pmap.parts[PartType.TOOL_CODE] = ["tools.py", "extras/"]
    (candidate.root / "extras").mkdir()
    (candidate.root / "extras" / "helper.py").write_text("x = 1\n")
    assert shell.is_strictly_additive(original, candidate)

    toy_fixes.fix_reverse(candidate.root)
    assert not shell.is_strictly_additive(original, candidate)
