"""Cold start: the coding agent GENERATES a runnable harness from a brief (no seed).

There are no templates — cold start is the same coding-agent engine that edits the
harness during hill-climbing, run on an empty workspace. In tests we script that
agent with a MockBackend ``agent_action`` that writes the files the agent would
write; the build loop validates the result (e.g. boot_check) and retries with the
error fed back, exactly like the edit loop.
"""

from studio.backends.mock import MockBackend
from studio.stages.optimize.edit.strategist import BUILD_TAG, build_harness
from studio.core.parts import PartMap, PartType
from studio.targets import ColdStartBrief, Target, ToolSpec, get_target, register

BRIEF = ColdStartBrief(
    domain="multi-hop web-search QA",
    io_contract="input: a question string; answer via finish(answer)",
    tools=[ToolSpec("search", "search(query: str) -> list[str]", "web search")],
    runner_contract="the runtime imports run_episode(task, call_model, tools) from agent.py",
)


def _writes(*files):
    """A scripted coding-agent action: write the given {path: content} files."""
    def action(workspace):
        for rel, content in files[0].items():
            (workspace / rel).parent.mkdir(parents=True, exist_ok=True)
            (workspace / rel).write_text(content)
    return action


def test_build_runs_the_coding_agent_and_returns_generated_harness(tmp_path):
    backend = MockBackend(agent_actions={BUILD_TAG: [
        _writes({"agent.py": "def run_episode(t, call_model, tools):\n    return ''\n"})]})
    h = build_harness(backend, tmp_path / "cold", BRIEF)
    assert ("run_agent", BUILD_TAG) in backend.calls       # the coding agent ran
    assert h.exists("agent.py")
    # The brief (domain + runner contract + tool) drove the instruction.
    _, instr = backend.prompt_log[0]
    assert "FROM SCRATCH" in instr and "multi-hop web-search QA" in instr
    assert "run_episode" in instr and "search" in instr


def test_build_seed_files_are_pre_dropped(tmp_path):
    backend = MockBackend(agent_actions={BUILD_TAG: [lambda ws: None]})  # agent adds nothing
    brief = ColdStartBrief(domain="d", io_contract="io", runner_contract="reads policy.md",
                           seed_files={"policy.md": "# starter\n"})
    h = build_harness(backend, tmp_path / "cold", brief)
    assert h.read_file("policy.md") == "# starter\n"


def test_build_validates_and_retries_until_it_boots(tmp_path):
    # First attempt writes a broken (empty) file -> validate fails -> second
    # attempt fixes it. The agent decides how to fix, given the error.
    backend = MockBackend(agent_actions={BUILD_TAG: [
        _writes({"policy.md": ""}),                 # attempt 1: empty -> invalid
        _writes({"policy.md": "real policy\n"}),    # attempt 2: valid
    ]})

    def validate(h):
        ok = h.exists("policy.md") and h.read_file("policy.md").strip() != ""
        return (ok, "" if ok else "policy.md is empty")

    h = build_harness(backend, tmp_path / "cold", BRIEF, validate=validate, max_attempts=2)
    assert h.read_file("policy.md") == "real policy\n"
    assert sum(1 for k, t in backend.calls if t == BUILD_TAG) == 2   # retried once
    # the retry instruction carried the validation error back to the agent
    assert "policy.md is empty" in backend.prompt_log[1][1]


def test_build_stops_at_first_success(tmp_path):
    backend = MockBackend(agent_actions={BUILD_TAG: [_writes({"policy.md": "ok\n"})]})
    build_harness(backend, tmp_path / "cold", BRIEF,
                  validate=lambda h: (True, ""), max_attempts=3)
    assert sum(1 for k, t in backend.calls if t == BUILD_TAG) == 1   # no needless retries


def test_target_cold_resolve_uses_the_builder(tmp_path):
    register("demo-cold", lambda: Target(
        name="demo-cold",
        make_benchmark=lambda cfg: None,
        part_map=lambda: PartMap(parts={PartType.INSTRUCTIONS: ["policy.md"]}, do_not_touch=[]),
        seed_harness=lambda: None,                 # cold start
        cold_start_brief=lambda: BRIEF,
        baseline_score=0.5,
    ))
    t = get_target("demo-cold")
    assert t.seed_harness() is None
    backend = MockBackend(agent_actions={BUILD_TAG: [_writes({"policy.md": "p\n"})]})
    seed = t.resolve_seed(backend, tmp_path / "ws", validate=lambda h: (True, ""))
    assert seed.exists("policy.md")


def test_warm_target_uses_seed_not_cold_start(tmp_path):
    from studio.benchmark.toy import build_toy_harness, toy_part_map

    src = build_toy_harness(tmp_path / "toy_src")
    register("demo-warm", lambda: Target(
        name="demo-warm", make_benchmark=lambda cfg: None,
        part_map=toy_part_map, seed_harness=lambda: src, baseline_score=0.25))
    # No backend needed: warm start copies the shipped seed (never runs the agent).
    seed = get_target("demo-warm").resolve_seed(None, tmp_path / "ws2")
    assert seed.exists("tools.py")
