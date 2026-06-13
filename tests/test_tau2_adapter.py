"""tau2-bench adapter: docker-free, policy-as-harness, clean per-candidate isolation."""

import json
from pathlib import Path

import pytest

from studio.benchmark.kira import BenchmarkExecutionError
from studio.benchmark.tau2 import (
    Tau2Benchmark, tau2_part_map, tau2_seed_harness,
)
from studio.harness import Harness
from studio.parts import PartType


def _fake_repo(tmp_path):
    """A minimal tau2 repo layout: data/tau2/{shared.json, domains/{airline,retail}/*}."""
    root = tmp_path / "tau2-bench"
    tau2 = root / "data" / "tau2"
    (tau2).mkdir(parents=True)
    (tau2 / "shared.json").write_text("{}")  # non-domain entry (import-safety symlink)
    for dom in ("airline", "retail"):
        d = tau2 / "domains" / dom
        d.mkdir(parents=True)
        (d / "policy.md").write_text(f"ORIGINAL {dom} policy\n")
        (d / "db.json").write_text('{"rows": []}')
        (d / "tasks.json").write_text(json.dumps([{"id": "t1"}, {"id": "t2"}, {"id": "t3"}]))
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "tau2").write_text("")  # bin must .exists()
    return root


def test_list_tasks_reads_domain_tasks(tmp_path):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", tau2_repo=repo)
    assert b.list_tasks() == ["t1", "t2", "t3"]


def test_build_cmd_has_correct_flags(tmp_path):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", model="gpt-4.1", user_model="gpt-4.1-mini",
                      k=3, n_concurrent=6, tau2_repo=repo)
    cmd = b.build_cmd(["t1", "t2"], Path("/tmp/out.json"), run_idx=2)
    s = " ".join(cmd)
    assert "run --domain airline" in s
    assert "--agent-llm gpt-4.1" in s and "--user-llm gpt-4.1-mini" in s
    assert "--num-trials 3" in s and "--max-concurrency 6" in s and "--seed 2" in s
    assert "--task-ids t1 t2" in s  # nargs="+" space-separated
    assert "--save-to /tmp/out.json" in s


def test_data_dir_overlay_isolates_mutated_policy(tmp_path):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", tau2_repo=repo)
    # harness with a MUTATED policy
    hroot = tmp_path / "harness"; hroot.mkdir()
    h = Harness(hroot); h.write_file("policy.md", "MUTATED airline policy v2\n")

    dest = tmp_path / "candidate_data"
    b._build_data_dir(h, dest)

    # our domain's policy.md is a REAL file with the mutated content...
    mine = dest / "tau2" / "domains" / "airline" / "policy.md"
    assert mine.is_file() and not mine.is_symlink()
    assert mine.read_text() == "MUTATED airline policy v2\n"
    # ...the original is untouched (isolation)
    assert (repo / "data" / "tau2" / "domains" / "airline" / "policy.md").read_text() == "ORIGINAL airline policy\n"
    # db.json/tasks.json are symlinked (not copied) for our domain
    assert (dest / "tau2" / "domains" / "airline" / "db.json").is_symlink()
    # the OTHER domain is symlinked wholesale
    assert (dest / "tau2" / "domains" / "retail").is_symlink()
    # non-domain shared data symlinked for import-safety
    assert (dest / "tau2" / "shared.json").is_symlink()


def test_run_parses_per_task_mean_reward(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", real=True, k=2, tau2_repo=repo)
    hroot = tmp_path / "h"; hroot.mkdir(); h = Harness(hroot); h.write_file("policy.md", "p\n")

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--save-to") + 1])
        # t1: 1.0 & 1.0 -> 1.0 ; t2: 1.0 & 0.0 -> 0.5 ; t3: 0.0 & 0.0 -> 0.0
        sims = [
            {"task_id": "t1", "trial": 0, "reward_info": {"reward": 1.0}, "messages": []},
            {"task_id": "t1", "trial": 1, "reward_info": {"reward": 1.0}},
            {"task_id": "t2", "trial": 0, "reward_info": {"reward": 1.0}},
            {"task_id": "t2", "trial": 1, "reward_info": {"reward": 0.0},
             "messages": [{"role": "assistant", "content": "wrong move"}]},
            {"task_id": "t3", "trial": 0, "reward_info": {"reward": 0.0}},
            {"task_id": "t3", "trial": 1, "reward_info": {"reward": 0.0}},
        ]
        out.write_text(json.dumps({"simulations": sims}))

        class R:
            returncode = 0
        return R()

    monkeypatch.setattr("studio.benchmark.tau2.subprocess.run", fake_run)
    scores = b.run(h, ["t1", "t2", "t3"], run_idx=0)
    assert scores == {"t1": 1.0, "t2": 0.5, "t3": 0.0}
    # a failing task captured a trace, scoped to this harness
    assert "wrong move" in b.last_trace("t2", harness=h)
    assert b.last_trace("t1", harness=h) == ""  # passed -> no excerpt
    # structured evidence is also available, scoped to the same harness
    assert b.last_evidence("t2", harness=h).reward == 0.0
    assert b.last_evidence("t1", harness=h) is None


def test_build_evidence_localizes_failed_action_mid_transcript(tmp_path):
    """A failed action_check naming a tool used mid-dialogue is localized to
    that turn (not the blind tail)."""
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", real=True, tau2_repo=repo)
    sim = {
        "task_id": "t1", "trial": 0,
        "reward_info": {
            "reward": 0.0,
            "action_checks": [
                {"action": {"name": "update_reservation", "arguments": {"id": "X"}},
                 "action_match": False, "action_reward": 0.0},
                {"action": {"name": "get_user_details", "arguments": {}},
                 "action_match": True, "action_reward": 1.0},  # passed -> not a signal
            ],
            "db_check": {"db_match": False, "db_reward": 0.0},
        },
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "change my flight"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "update_reservation", "arguments": {"id": "X"}, "requestor": "assistant"}]},
            {"role": "tool", "content": "error: cannot modify"},
            {"role": "assistant", "content": "sorry"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "goodbye"},
        ],
    }
    ev = b._build_evidence(sim)
    assert ev.reward == 0.0
    # the failed action signal is present; the passed one is too but marked passed
    failed = [s for s in ev.signals if not s.passed]
    assert any(s.name == "update_reservation" for s in failed)
    assert any(s.kind == "db" for s in failed)
    # a window covers the update_reservation call site (index 2), not just the tail
    covered = {i for w in ev.windows for i in range(w.start_idx, w.end_idx + 1)}
    assert 2 in covered
    assert ev.transcript_len == 7 and ev.full_messages  # full transcript retained


def test_build_evidence_graceful_on_malformed_reward_info(tmp_path):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", real=True, tau2_repo=repo)
    sim = {"task_id": "t1", "reward_info": {"reward": 0.0, "action_checks": "garbage"},
           "messages": [{"role": "assistant", "content": "last word"}]}
    ev = b._build_evidence(sim)  # must not raise
    assert ev.windows and "last word" in ev.windows[-1].messages[-1]["content"]


def test_parse_results_keeps_worst_failing_trial(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", real=True, k=2, tau2_repo=repo)
    hroot = tmp_path / "h"; hroot.mkdir(); h = Harness(hroot); h.write_file("policy.md", "p\n")

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--save-to") + 1])
        out.write_text(json.dumps({"simulations": [
            {"task_id": "t1", "trial": 0, "reward_info": {"reward": 0.5},
             "messages": [{"role": "assistant", "content": "mild miss"}]},
            {"task_id": "t1", "trial": 1, "reward_info": {"reward": 0.0},
             "messages": [{"role": "assistant", "content": "total miss"}]},
        ]}))
        class R: returncode = 0
        return R()

    monkeypatch.setattr("studio.benchmark.tau2.subprocess.run", fake_run)
    scores = b.run(h, ["t1"], run_idx=0)
    assert scores == {"t1": 0.25}                      # mean over trials
    ev = b.last_evidence("t1", harness=h)
    assert ev.reward == 0.0 and "total miss" in b.last_trace("t1", harness=h)


def test_run_raises_on_missing_results(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    b = Tau2Benchmark(domain="airline", real=True, tau2_repo=repo)
    hroot = tmp_path / "h"; hroot.mkdir(); h = Harness(hroot); h.write_file("policy.md", "p\n")
    monkeypatch.setattr("studio.benchmark.tau2.subprocess.run",
                        lambda cmd, **kw: type("R", (), {"returncode": 1})())
    with pytest.raises(BenchmarkExecutionError, match="rc=1"):
        b.run(h, ["t1"], run_idx=0)


def test_editable_instruction_expands_the_surface(tmp_path):
    """With the agent-instruction lever on, the harness has TWO editable levers
    (policy + agent_instruction), the run injects the mutated instruction via
    TAU2_AGENT_INSTRUCTION, and boot_check requires both."""
    repo = _fake_repo(tmp_path)
    # simulate a patched tau2 source so auto-detect would also fire
    agentpy = repo / "src" / "tau2" / "agent" / "llm_agent.py"
    agentpy.parent.mkdir(parents=True, exist_ok=True)
    agentpy.write_text("AGENT_INSTRUCTION = os.environ.get('TAU2_AGENT_INSTRUCTION') or 'default'\n")
    from studio.benchmark.tau2 import instruction_injectable, AGENT_INSTRUCTION_FILE
    assert instruction_injectable(repo) is True

    b = Tau2Benchmark(domain="airline", real=True, tau2_repo=repo, editable_instruction=True)
    pm = tau2_part_map("airline", editable_instruction=True)
    assert pm.files_for(PartType.INSTRUCTIONS) == ["policy.md", AGENT_INSTRUCTION_FILE]

    seed = tau2_seed_harness("airline", tmp_path / "seed", tau2_repo=repo, editable_instruction=True)
    assert seed.exists("policy.md") and seed.exists(AGENT_INSTRUCTION_FILE)
    assert b.boot_check(seed)[0] is True
    # missing the instruction file -> boot_check fails when the lever is on
    pol_only = Harness(tmp_path / "po"); (tmp_path / "po").mkdir(); pol_only.write_file("policy.md", "p")
    assert b.boot_check(pol_only)[0] is False

    # run() injects the mutated instruction into the env
    h = Harness(tmp_path / "h"); (tmp_path / "h").mkdir()
    h.write_file("policy.md", "p"); h.write_file(AGENT_INSTRUCTION_FILE, "MUTATED INSTRUCTION v2")
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw.get("env", {}))
        out = Path(cmd[cmd.index("--save-to") + 1])
        out.write_text(json.dumps({"simulations": [
            {"task_id": "t1", "reward_info": {"reward": 1.0}}]}))
        class R: returncode = 0
        return R()

    import studio.benchmark.tau2 as tau2mod
    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    monkey.setattr(tau2mod.subprocess, "run", fake_run)
    b.run(h, ["t1"], run_idx=0)
    monkey.undo()
    assert seen.get("TAU2_AGENT_INSTRUCTION") == "MUTATED INSTRUCTION v2"


def test_seed_harness_and_part_map(tmp_path):
    repo = _fake_repo(tmp_path)
    seed = tau2_seed_harness("airline", tmp_path / "seed", tau2_repo=repo)
    assert seed.read_file("policy.md") == "ORIGINAL airline policy\n"
    pm = tau2_part_map("airline", editable_instruction=False)
    assert pm.files_for(PartType.INSTRUCTIONS) == ["policy.md"]
    # boot_check rejects an empty policy, accepts a real one
    b = Tau2Benchmark(domain="airline", tau2_repo=repo)
    assert b.boot_check(seed)[0] is True
    empty = Harness(tmp_path / "e"); (tmp_path / "e").mkdir(); empty.write_file("policy.md", "  ")
    assert b.boot_check(empty)[0] is False
