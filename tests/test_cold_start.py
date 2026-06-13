"""Cold start: synthesize a runnable harness from a ColdStartBrief (no seed).

This is the capability that lets SHO hill-climb a benchmark that ships no agent
harness (e.g. BrowseComp): the optimizer's round-0 harness is generated, not
provided."""

import json

from studio import schemas
from studio.backends.mock import MockBackend
from studio.components.cold_start import (
    bootstrap_harness, cold_start_part_map,
)
from studio.parts import PartType
from studio.targets import ColdStartBrief, Target, TargetConfig, ToolSpec, get_target, register


BRIEF = ColdStartBrief(
    domain="multi-hop web-search QA",
    io_contract="input: a question string; answer via finish(answer)",
    tools=[
        ToolSpec("search", "search(query: str) -> list[str]", "web search, returns snippets"),
        ToolSpec("open", "open(url: str) -> str", "fetch page text"),
        ToolSpec("finish", "finish(answer: str) -> None", "submit the final answer"),
    ],
    template="react",
)

SYNTH = {
    "system_prompt": "You answer questions by searching the web. Use search and open, "
                     "then call finish(answer).",
    "tool_notes": [
        {"name": "search", "note": "issue focused queries"},
        {"name": "finish", "note": "call exactly once with the answer"},
    ],
    "loop_guidance": "Plan, search, read, verify, then finish.",
}


def test_bootstrap_produces_runnable_react_harness(tmp_path):
    backend = MockBackend(json_responses={"cold-start": [SYNTH]})
    h = bootstrap_harness(backend, BRIEF, tmp_path / "cold")

    files = set(h.files())
    assert files == {"system_prompt.md", "tools.md", "tool_schemas.json", "agent.py", "config.json"}
    # The synthesized prompt landed.
    assert "searching the web" in h.read_file("system_prompt.md")
    # Tool descriptions include each tool + the per-tool note.
    tools_md = h.read_file("tools.md")
    assert "search" in tools_md and "open" in tools_md and "finish" in tools_md
    assert "issue focused queries" in tools_md
    assert "Plan, search, read" in tools_md  # loop guidance folded in
    # tool_schemas.json is valid JSON listing all 3 tools.
    schema = json.loads(h.read_file("tool_schemas.json"))
    assert {t["name"] for t in schema} == {"search", "open", "finish"}
    # agent.py is valid Python with a run_episode loop and a real step cap.
    agent = h.read_file("agent.py")
    assert "def run_episode" in agent and "MAX_STEPS = 12" in agent
    compile(agent, "agent.py", "exec")
    # config.json parses with the step budget.
    assert json.loads(h.read_file("config.json"))["max_steps"] == 12


def test_synthesis_failure_degrades_to_a_minimal_but_valid_harness(tmp_path):
    # Model returns an empty/sparse object -> still produces a runnable harness
    # (the optimizer can climb from a weak baseline; it must not crash).
    backend = MockBackend(json_responses={"cold-start": [{"system_prompt": "", "tool_notes": []}]})
    h = bootstrap_harness(backend, BRIEF, tmp_path / "cold")
    assert "multi-hop web-search QA" in h.read_file("system_prompt.md")  # fallback prompt
    compile(h.read_file("agent.py"), "agent.py", "exec")


def test_cold_start_part_map_matches_template_files(tmp_path):
    pm = cold_start_part_map("react")
    assert pm.files_for(PartType.INSTRUCTIONS) == ["system_prompt.md"]
    assert "tool_schemas.json" in pm.files_for(PartType.TOOL_DESCRIPTIONS)
    assert "agent.py" in pm.files_for(PartType.MIDDLEWARE)
    # Every mapped file is one the bootstrapper actually writes.
    backend = MockBackend(json_responses={"cold-start": [SYNTH]})
    h = bootstrap_harness(backend, BRIEF, tmp_path / "cold")
    written = set(h.files())
    for part in (PartType.INSTRUCTIONS, PartType.TOOL_DESCRIPTIONS, PartType.MIDDLEWARE):
        for f in pm.files_for(part):
            assert f in written


def test_target_registry_and_cold_resolve(tmp_path):
    # A registered cold-start target resolves a synthesized seed (no shipped harness).
    register("demo-cold", lambda: Target(
        name="demo-cold",
        make_benchmark=lambda cfg: None,
        part_map=lambda: cold_start_part_map("react"),
        seed_harness=lambda: None,                 # cold start
        cold_start_brief=lambda: BRIEF,
        baseline_score=0.5,
    ))
    t = get_target("demo-cold")
    assert t.seed_harness() is None
    backend = MockBackend(json_responses={"cold-start": [SYNTH]})
    seed = t.resolve_seed(backend, tmp_path / "ws")
    assert seed.exists("agent.py")
    assert "searching the web" in seed.read_file("system_prompt.md")


def test_warm_target_uses_seed_not_cold_start(tmp_path):
    from studio.benchmark.toy import build_toy_harness, toy_part_map

    src = build_toy_harness(tmp_path / "toy_src")
    register("demo-warm", lambda: Target(
        name="demo-warm",
        make_benchmark=lambda cfg: None,
        part_map=toy_part_map,
        seed_harness=lambda: src,
        baseline_score=0.25,
    ))
    t = get_target("demo-warm")
    # No backend needed: warm start copies the shipped seed.
    seed = t.resolve_seed(None, tmp_path / "ws2")
    assert seed.exists("tools.py")  # the toy harness, not a cold-start react harness
    assert not seed.exists("agent.py")


def test_cold_start_schema_validates():
    schemas.validate(SYNTH, schemas.COLD_START)
    try:
        schemas.validate({"tool_notes": []}, schemas.COLD_START)  # missing system_prompt
        raise AssertionError("should require system_prompt")
    except schemas.SchemaError:
        pass
