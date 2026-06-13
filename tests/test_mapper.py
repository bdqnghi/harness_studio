from studio.backends.mock import MockBackend
from studio.benchmark.toy import build_toy_harness
from studio.stages.optimize.edit import mapper
from studio.core.parts import PartType


def test_map_harness_parses_and_restricts(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    # Mapper returns a list/absent per part; "ghost.py" doesn't exist and must be dropped.
    response = {
        "instructions": ["instructions.txt"],
        "tool_code": ["tools.py", "ghost.py"],
        "tool_descriptions": "absent",
        "middleware": ["config.json"],
        "skills": "absent",
        "subagents": "absent",
        "memory": "absent",
        "do_not_touch": [],
    }
    backend = MockBackend(json_responses={"mapper": [response]})
    result = mapper.map_harness(backend, h)
    assert result.files_for(PartType.INSTRUCTIONS) == ["instructions.txt"]
    assert result.files_for(PartType.TOOL_CODE) == ["tools.py"]  # ghost.py dropped
    assert not result.is_present(PartType.SKILLS)


def test_build_listing_includes_tree_and_heads(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    listing = mapper.build_listing(h)
    assert "tools.py" in listing and "OPS = {" in listing
