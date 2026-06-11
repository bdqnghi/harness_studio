"""Directory-aware part map: the fairness fix that lets the optimizer ADD files
(new tools/middleware/skills) under a mapped directory, like AHE can — not just
edit pre-listed files. Locks in studio/parts.py + studio/components/shell.py."""

from studio.components import shell
from studio.harness import Harness
from studio.parts import PartMap, PartType
from studio.benchmark.nexau import api_type_for


def _harness(tmp_path):
    root = tmp_path / "h"
    (root / "tools").mkdir(parents=True)
    (root / "systemprompt.md").write_text("you are an agent\n")
    (root / "tools" / "shell.py").write_text("def run(): ...\n")
    (root / "nexau.json").write_text("{}\n")
    return Harness(root)


def _dir_map():
    return PartMap(
        parts={
            PartType.INSTRUCTIONS: ["systemprompt.md"],
            PartType.TOOL_CODE: ["tools/"],          # directory part
        },
        do_not_touch=["nexau.json"],
    )


def test_dir_entry_editability_and_bucketing():
    pm = _dir_map()
    assert pm.is_editable("tools/shell.py")          # existing under dir
    assert pm.is_editable("tools/new_tool.py")       # NEW under dir
    assert pm.is_editable("systemprompt.md")         # exact file
    assert not pm.is_editable("nexau.json")          # do-not-touch
    assert pm.part_of("tools/new_tool.py") is PartType.TOOL_CODE
    assert pm.part_of("systemprompt.md") is PartType.INSTRUCTIONS
    assert pm.part_of("nexau.json") is None


def test_shell_keeps_new_file_under_dir_part(tmp_path):
    original = _harness(tmp_path)
    candidate = original.copy_to(tmp_path / "cand")
    (candidate.root / "tools" / "file_edit.py").write_text("def edit(): ...\n")  # ADD a tool
    res = shell.enforce(original, candidate, _dir_map(), budget_per_part=3)
    assert res.ok
    assert "tools/file_edit.py" in res.changed_parts[PartType.TOOL_CODE]
    assert (candidate.root / "tools" / "file_edit.py").exists()  # NOT reverted


def test_shell_reverts_new_file_outside_any_part(tmp_path):
    original = _harness(tmp_path)
    candidate = original.copy_to(tmp_path / "cand")
    (candidate.root / "secret.txt").write_text("exfil\n")        # outside mapped dirs
    res = shell.enforce(original, candidate, _dir_map(), budget_per_part=3)
    assert "secret.txt" in res.reverted
    assert not (candidate.root / "secret.txt").exists()


def test_dir_part_respects_budget(tmp_path):
    original = _harness(tmp_path)
    candidate = original.copy_to(tmp_path / "cand")
    for i in range(3):
        (candidate.root / "tools" / f"t{i}.py").write_text("x = 1\n")
    res = shell.enforce(original, candidate, _dir_map(), budget_per_part=2)
    assert not res.ok
    assert res.violations and "tool_code" in res.violations[0]


def test_nexau_provider_is_explicit_and_validated():
    assert api_type_for("custom-name", "openai") == "openai_responses"
    assert api_type_for("custom-name", "gemini") == "gemini_rest"
