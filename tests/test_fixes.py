"""Regression tests for the bugs found by the adversarial review workflow."""

import pytest

from studio import schemas
from studio.backends._jsonio import extract_json as _extract_json
from studio.backends.mock import MockBackend
from studio.benchmark.base import Benchmark
from studio.benchmark.toy import FAMILIES, build_toy_harness, toy_part_map
from studio.stages.optimize import mapper
from studio.stages.split import TaskSplit
from studio.config import Config, EditConfig, LoopConfig
from studio.stages.optimize.orchestrator import Orchestrator
from studio.core.parts import PartMap, PartType


# --- files_changed must include deleted files ---

def test_files_changed_reports_deletions(tmp_path):
    h = build_toy_harness(tmp_path / "h")

    def delete_config(root):
        (root / "config.json").unlink()

    be = MockBackend(agent_actions={"strategist": [delete_config]})
    res = be.run_agent("go", workspace=h.root, tag="strategist")
    assert "config.json" in res.files_changed


# --- PartMap.from_dict tolerates malformed AI output (no crash) ---

@pytest.mark.parametrize("bad", [True, 123, {"nested": "x"}, None, "absent"])
def test_partmap_from_dict_coerces_bad_values(bad):
    pm = PartMap.from_dict({"instructions": bad, "do_not_touch": []})
    assert pm.files_for(PartType.INSTRUCTIONS) == []  # never crashes, never junk


def test_partmap_from_dict_bare_string_becomes_singleton():
    pm = PartMap.from_dict({"tool_code": "tools.py", "do_not_touch": []})
    assert pm.files_for(PartType.TOOL_CODE) == ["tools.py"]


# --- the schema now rejects a non-array, non-"absent" part value ---

def test_part_map_schema_rejects_bool_part():
    bad = {"instructions": True, "tool_descriptions": "absent", "tool_code": [],
           "middleware": [], "skills": "absent", "subagents": "absent",
           "memory": "absent", "do_not_touch": []}
    with pytest.raises(schemas.SchemaError):
        schemas.validate(bad, schemas.PART_MAP)


def test_part_map_schema_accepts_array_or_absent():
    good = {"instructions": ["a.txt"], "tool_descriptions": "absent",
            "tool_code": ["t.py"], "middleware": "absent", "skills": "absent",
            "subagents": "absent", "memory": "absent", "do_not_touch": ["x"]}
    schemas.validate(good, schemas.PART_MAP)  # must not raise


# --- Mapper resolves a file mapped to both a part and do_not_touch ---

def test_mapper_drops_do_not_touch_overlap(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    resp = {"instructions": ["instructions.txt"], "tool_code": ["tools.py"],
            "tool_descriptions": "absent", "middleware": "absent", "skills": "absent",
            "subagents": "absent", "memory": "absent",
            "do_not_touch": ["tools.py", "config.json"]}  # tools.py overlaps tool_code
    pm = mapper.map_harness(MockBackend(json_responses={"mapper": [resp]}), h)
    assert "tools.py" not in pm.do_not_touch       # overlap removed
    assert "config.json" in pm.do_not_touch         # genuine do-not-touch kept


# --- _extract_json tolerates trailing prose (and is O(n)) ---

def test_extract_json_with_trailing_prose():
    assert _extract_json('{"a": 1} and that is my answer') == {"a": 1}


def test_extract_json_with_code_fence():
    assert _extract_json('```json\n[1, 2, 3]\n```') == [1, 2, 3]


# --- final-segment deep audit catches a trap that passed the fast gate ---

class TrapBench(Benchmark):
    """A '# TRAP' edit looks great on judging/final but is secretly worse on the
    audit pile — exactly the case the deep auditor must catch on the last segment."""

    TASKS = [f"{p}{i}" for p in ("prac", "judge", "aud", "fin") for i in range(4)]

    def list_tasks(self):
        return list(self.TASKS)

    def run(self, harness, task_ids, *, run_idx=0):
        hacked = "# TRAP" in harness.read_file("tools.py")
        out = {}
        for t in task_ids:
            if t.startswith("aud"):
                out[t] = 0.0 if hacked else 1.0   # trap regresses the audit pile
            else:
                out[t] = 1.0 if hacked else 0.0   # trap inflates everything else
        return out

    def boot_check(self, harness):
        return True, ""


def inject_trap(root):
    p = root / "tools.py"
    p.write_text(p.read_text() + "\n# TRAP\n")


GOOD_TOOLS = (
    "def _echo(a):\n    return a\n\n\n"
    "def _reverse(a):\n    return a[::-1]\n\n\n"
    "def _upper(a):\n    return a.upper()\n\n\n"
    'OPS = {"echo": _echo, "reverse": _reverse, "upper": _upper}\n'
)


def test_repair_touching_do_not_touch_is_reverted(tmp_path):
    """A structural-repair that also edits a do-not-touch file must have that
    edit reverted by re-enforcing the shell before the gate (fix B)."""
    from studio.benchmark.toy import ToyBenchmark
    from studio.benchmark import toy_fixes as fixes

    split = TaskSplit(
        held_in=[f"{f}-{i}" for f in FAMILIES for i in (0, 1, 4, 5, 6, 7)],
        held_out=[f"{f}-{i}" for f in FAMILIES for i in (2, 3)],
    )
    pmap = toy_part_map()
    pmap.parts[PartType.MIDDLEWARE] = []           # config.json no longer editable
    pmap.do_not_touch = ["config.json"]            # ... it is do-not-touch

    def repair(root):  # fix the boot AND illegally touch a do-not-touch file
        (root / "tools.py").write_text(GOOD_TOOLS)
        (root / "config.json").write_text('{"HACKED": true}\n')

    backend = MockBackend(
        json_responses={
            "diagnoser": [[{"pattern_id": "p", "description": "x", "root_cause": "x",
                            "failing_task_ids": ["reverse-0"], "blamed_part": "tool_code",
                            "confidence": 0.5, "addressable": True}]],
            "direction-router": [{"assignments": [{"pattern_id": "p", "direction_id": "",
                                  "new_title": "fix ops", "new_mechanism": "ops buggy"}]}],
            "ideator": [{"hypotheses": [{"title": "fix reverse", "mechanism": "m",
                         "hypothesis": "make reverse correct", "observable": "reverse passes"}]}],
            "insight": [{"insight": "fixing the op works"}],
            "insight-direction": [{"insight": "ops were the bottleneck"}],
        },
        agent_actions={"strategist": [fixes.break_boot], "strategist-repair": [repair]},
    )
    config = Config(loop=LoopConfig(rounds=1, segment_length=10, wobble_runs=2),
                    edits=EditConfig(allow_repair=True))
    src = build_toy_harness(tmp_path / "src")
    original_config = src.read_file("config.json")
    orch = Orchestrator(
        workspace=tmp_path / "ws", source_harness=src,
        benchmark=ToyBenchmark(per_family=12), backend=backend, config=config,
        split=split, part_map=pmap,
    )
    result = orch.run()

    assert result.accepted == 1                              # repaired edit reached the gate
    assert "reverse" in orch.harness.read_file("tools.py")   # the real fix landed
    assert orch.harness.read_file("config.json") == original_config  # do-not-touch reverted


