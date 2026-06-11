"""mini-swe-agent target: part map shape + editability + the real=False guard.

Locks in studio/benchmark/mini_swe.py. No Docker, no network — the run() guard
must raise before anything is invoked.
"""

import pytest

from studio.benchmark.mini_swe import MiniSweBenchmark, mini_swe_part_map
from studio.harness import Harness
from studio.parts import PartMap, PartType


def test_part_map_is_a_partmap_with_expected_parts():
    pm = mini_swe_part_map()
    assert isinstance(pm, PartMap)
    # The harness exposes instructions, tool code, and memory at minimum.
    assert pm.is_present(PartType.INSTRUCTIONS)
    assert pm.is_present(PartType.TOOL_CODE)
    assert pm.is_present(PartType.MEMORY)
    # SKILLS / SUBAGENTS are absent in this harness.
    assert not pm.is_present(PartType.SKILLS)
    assert not pm.is_present(PartType.SUBAGENTS)


def test_editability_under_mapped_dirs_and_do_not_touch():
    pm = mini_swe_part_map()
    # A new/existing config YAML lives under the mapped INSTRUCTIONS dir.
    assert pm.is_editable("src/minisweagent/config/mini.yaml")
    # Inert infra is off-limits.
    assert not pm.is_editable("pyproject.toml")
    # Directory parts let the optimizer ADD files under them.
    assert pm.is_editable("src/minisweagent/config/new_preset.yaml")
    assert pm.part_of("src/minisweagent/config/mini.yaml") is PartType.INSTRUCTIONS
    assert pm.part_of("pyproject.toml") is None


def test_run_raises_when_not_real(tmp_path):
    root = tmp_path / "h"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    bench = MiniSweBenchmark(real=False)
    with pytest.raises(NotImplementedError):
        bench.run(Harness(root), ["some-task"])


def test_list_tasks_empty_when_not_real():
    assert MiniSweBenchmark(real=False).list_tasks() == []


def test_explicit_tasks_returned():
    assert MiniSweBenchmark(real=False, tasks=["a", "b"]).list_tasks() == ["a", "b"]


def test_build_cmd_has_agent_and_no_config_dir(tmp_path):
    root = tmp_path / "h"
    root.mkdir()
    bench = MiniSweBenchmark(real=True, model="gemini/gemini-3.5-flash")
    cmd = bench.build_cmd(Harness(root), ["t1"], tmp_path / "jobs", tmp_path / "ds")
    assert "--agent" in cmd
    assert cmd[cmd.index("--agent") + 1] == "mini-swe-agent"
    # Stock harbor mini-swe-agent command takes no config dir.
    assert "--ak" not in cmd
    # Model is passed through in litellm format.
    assert cmd[cmd.index("--model") + 1] == "gemini/gemini-3.5-flash"
