"""tau2-bench adapter — a FAST, docker-free hill-climb target.

Unlike the harbor/docker adapters (nexau, mini_swe, kira), tau2-bench is pure
Python: each task is a multi-turn tool-use dialogue between the agent under test
and an LLM user-simulator, scored by a deterministic DB-state-diff + action
verifier. No container, no per-rollout reinstall tax — one rollout is seconds to
~1 min of LLM round-trips.

THE EDITABLE HARNESS (what SHO hill-climbs): the domain **policy** — the bulk of
what the agent reads and the dominant lever (tau2's own paper ablates rewriting
it and shows it moves scores). We keep it as a one-file harness ``policy.md``
mapped to INSTRUCTIONS. Per-candidate isolation is clean and parallel-safe: we
build a throwaway ``TAU2_DATA_DIR`` symlink-farm of the original data and overlay
only the mutated ``policy.md``, so candidates never collide and tau2's source is
never edited. (Editing ``AGENT_INSTRUCTION`` too is a phase-2 add via a custom
agent plugin.)

Scoring: run ``tau2 run ... --save-to out.json`` then average
``reward_info.reward`` over trials per task → per-task score in [0,1].
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..core.evidence import (
    EvidenceStore,
    TaskEvidence,
    VerifierSignal,
    select_windows,
    to_flat_excerpt,
)
from ..core.harness import Harness
from ..core.parts import PartMap, PartType
from .base import Benchmark
from .kira import BenchmarkExecutionError

DEFAULT_TAU2_REPO = Path(
    os.environ.get("TAU2_REPO", str(Path(__file__).resolve().parents[3] / "tau2-bench"))
)
POLICY_FILE = "policy.md"
# A second editable lever beyond the domain policy: the agent's behavioral
# instruction (tau2's AGENT_INSTRUCTION — the <instructions> block, distinct
# from the <policy>). Injected per-candidate via the TAU2_AGENT_INSTRUCTION env
# var, which the tau2 source reads ONLY once it has been patched to do so — see
# instruction_injectable(). Until then the adapter stays policy-only, so this
# expansion is inert (and safe) on an unpatched install.
AGENT_INSTRUCTION_FILE = "agent_instruction.txt"
DOMAINS = ("airline", "retail", "telecom")

# Editable policy file(s) per domain (the harness). airline/retail ship a single
# policy.md; telecom splits its policy, so we hill-climb its main_policy.md.
_POLICY_FILES = {
    "airline": ["policy.md"],
    "retail": ["policy.md"],
    "telecom": ["main_policy.md"],
}


def _short(value, cap: int = 120) -> str:
    return str(value)[:cap] if value is not None else ""


def _decode_signals(ri: dict) -> list[VerifierSignal]:
    """Turn tau2's ``reward_info`` dict into structured verifier signals.

    Decodes action_checks / nl_assertions / communicate_checks / db_check /
    env_assertions into :class:`VerifierSignal`s. Action checks carry the tool
    name (so the localizer can index the transcript to the call site); the
    others are unnamed (they rely on the terminal window). Never raises."""
    sigs: list[VerifierSignal] = []
    try:
        for ac in ri.get("action_checks") or []:
            if not isinstance(ac, dict):
                continue
            act = ac.get("action") if isinstance(ac.get("action"), dict) else {}
            name = str(act.get("name", ""))
            passed = bool(ac.get("action_match", True))
            detail = "" if passed else f"expected action {name}({_short(act.get('arguments'))})"
            sigs.append(VerifierSignal("action", name, passed, detail))
        for nl in ri.get("nl_assertions") or []:
            if not isinstance(nl, dict):
                continue
            passed = bool(nl.get("met", True))
            detail = "" if passed else _short(nl.get("nl_assertion", ""), 300)
            if not passed and nl.get("justification"):
                detail += f" — {_short(nl['justification'], 200)}"
            sigs.append(VerifierSignal("nl_assertion", "", passed, detail))
        for cc in ri.get("communicate_checks") or []:
            if not isinstance(cc, dict):
                continue
            passed = bool(cc.get("met", True))
            sigs.append(VerifierSignal(
                "communicate", "", passed,
                "" if passed else f"did not communicate: {_short(cc.get('info', ''), 200)}"))
        db = ri.get("db_check")
        if isinstance(db, dict):
            passed = bool(db.get("db_match", True))
            sigs.append(VerifierSignal(
                "db", "", passed,
                "" if passed else "final database state does not match the target"))
        for ea in ri.get("env_assertions") or []:
            if not isinstance(ea, dict):
                continue
            passed = bool(ea.get("met", True))
            sigs.append(VerifierSignal("other", "env_assertion", passed,
                                       "" if passed else "environment assertion failed"))
    except Exception:  # noqa: BLE001 — evidence decode must never break a run
        return sigs
    return sigs


def _slim_message(m: dict) -> dict:
    """A compact, audio-free copy of a tau2 message for the stored transcript
    (drops base64 audio/effects fields; keeps role, content, tool calls)."""
    out: dict = {"role": m.get("role", "?"), "content": str(m.get("content") or "")[:1000]}
    calls = m.get("tool_calls")
    if isinstance(calls, list):
        out["tool_calls"] = [
            {"name": c.get("name", ""), "arguments": _short(c.get("arguments"), 200)}
            for c in calls if isinstance(c, dict)
        ]
    if m.get("name"):
        out["name"] = m["name"]
    return out


def instruction_injectable(tau2_repo: Path = DEFAULT_TAU2_REPO) -> bool:
    """True iff tau2's llm_agent.py has been patched to read AGENT_INSTRUCTION
    from the TAU2_AGENT_INSTRUCTION env var. Auto-detected so the agent-
    instruction lever turns on exactly when the source supports it (no manual
    flag, and zero effect on an unpatched install)."""
    src = Path(tau2_repo) / "src" / "tau2" / "agent" / "llm_agent.py"
    try:
        return "TAU2_AGENT_INSTRUCTION" in src.read_text()
    except OSError:
        return False


def default_agent_instruction() -> str:
    """tau2's shipped AGENT_INSTRUCTION (the warm-start seed for that lever)."""
    try:
        from tau2.agent.llm_agent import AGENT_INSTRUCTION
        return AGENT_INSTRUCTION
    except Exception:  # noqa: BLE001 — tau2 not importable; fall back to a stub
        return ("You are a customer service agent. In each turn either send a "
                "message to the user OR make a tool call, not both. Follow the "
                "policy. Always generate valid JSON only.")


def policy_files_for(domain: str) -> list[str]:
    return _POLICY_FILES.get(domain, [POLICY_FILE])


class Tau2Benchmark(Benchmark):
    def __init__(
        self,
        *,
        domain: str = "airline",
        model: str = "gpt-4.1",
        user_model: str = "gpt-4.1-mini",
        k: int = 1,
        n_concurrent: int = 8,
        real: bool = False,
        tau2_repo: Path = DEFAULT_TAU2_REPO,
        max_steps: int = 30,
        tasks: list[str] | None = None,
        tau2_bin: Path | None = None,
        policy_files: list[str] | None = None,
        editable_instruction: bool | None = None,
    ) -> None:
        if domain not in DOMAINS:
            raise ValueError(f"unsupported tau2 domain {domain!r}; pick from {DOMAINS}")
        self.domain = domain
        self.policy_files = list(policy_files or policy_files_for(domain))
        # Auto-enable the agent-instruction lever only if the tau2 source can
        # consume it (patched). Explicit True/False overrides the auto-detect.
        self.editable_instruction = (
            instruction_injectable(tau2_repo) if editable_instruction is None
            else editable_instruction
        )
        self.model = model
        self.user_model = user_model
        self.k = k
        self.n_concurrent = n_concurrent
        self.real = real
        self.tau2_repo = Path(tau2_repo)
        self.max_steps = max_steps
        self.tasks = list(tasks or [])
        self.tau2_bin = Path(tau2_bin) if tau2_bin else self.tau2_repo / ".venv" / "bin" / "tau2"
        # Structured failure evidence, versioned per harness hash (replaces the
        # old flat-excerpt dict). The localizer + editor consume this.
        self.evidence_store = EvidenceStore()

    # --- paths ---

    def _orig_domain_dir(self) -> Path:
        return self.tau2_repo / "data" / "tau2" / "domains" / self.domain

    def seed_policy_path(self) -> Path:
        return self._orig_domain_dir() / self.policy_files[0]

    # --- task discovery ---

    def list_tasks(self) -> list[str]:
        if self.tasks:
            return list(self.tasks)
        # Use tau2's registry so the ids match exactly what the runner accepts.
        # The CLI's default task split is "base"; some domains (telecom) ship a
        # huge full set under tasks.json whose ids the "base" split rejects, so
        # reading tasks.json directly is wrong there. Registry is the source of
        # truth; fall back to tasks.json only if the registry is unavailable.
        try:
            from tau2.registry import registry  # noqa: PLC0415
            tasks = registry.get_tasks_loader(self.domain)(task_split_name="base")
            return [str(t.id) for t in tasks if getattr(t, "id", None)]
        except Exception:  # noqa: BLE001 — registry import/load issues -> fallback
            tasks_json = self._orig_domain_dir() / "tasks.json"
            if not tasks_json.is_file():
                return []
            data = json.loads(tasks_json.read_text())
            items = data if isinstance(data, list) else data.get("tasks", [])
            return [str(t.get("id")) for t in items if isinstance(t, dict) and t.get("id")]

    # --- per-candidate data dir (symlink farm + mutated policy) ---

    def _build_data_dir(self, harness: Harness, dest: Path) -> Path:
        """A throwaway TAU2_DATA_DIR mirroring the original data via symlinks,
        with ONLY our domain's policy.md replaced by the harness's. Parallel-safe
        and never touches the tau2 source."""
        orig = self.tau2_repo / "data" / "tau2"
        td = dest / "tau2"
        domains = td / "domains"
        domains.mkdir(parents=True, exist_ok=True)
        # symlink everything under tau2/ except domains (shared data, import-safety)
        for entry in orig.iterdir():
            if entry.name != "domains":
                (td / entry.name).symlink_to(entry)
        # symlink sibling domains; rebuild OUR domain with mutated policy
        for d in (orig / "domains").iterdir():
            if d.name != self.domain:
                (domains / d.name).symlink_to(d)
        mine = domains / self.domain
        mine.mkdir()
        editable = set(self.policy_files)
        for f in self._orig_domain_dir().iterdir():
            if f.name in editable:
                continue  # overlay a mutated copy below
            (mine / f.name).symlink_to(f)
        for pf in self.policy_files:
            content = harness.read_file(pf) if harness.exists(pf) else (self._orig_domain_dir() / pf).read_text()
            (mine / pf).write_text(content)
        return dest

    # --- command construction (pure; used by dry-runs and run) ---

    def build_cmd(self, task_ids: list[str], out_path: Path, run_idx: int) -> list[str]:
        cmd = [
            str(self.tau2_bin), "run",
            "--domain", self.domain,
            "--agent-llm", self.model,
            "--user-llm", self.user_model,
            "--num-trials", str(max(1, self.k)),
            "--max-steps", str(self.max_steps),
            "--max-concurrency", str(self.n_concurrent),
            "--seed", str(run_idx),
            "--save-to", str(out_path),
            "--task-ids", *task_ids,
        ]
        return cmd

    # --- scoring ---

    def run(self, harness: Harness, task_ids, *, run_idx: int = 0) -> dict[str, float]:
        if not self.real:
            raise NotImplementedError(
                "real tau2 scoring is opt-in (real=True) and needs the tau2 CLI + model creds"
            )
        task_ids = list(task_ids)
        if not task_ids:
            return {}
        if not self.tau2_bin.exists():
            # Fall back to a tau2 on PATH, or — more reliably — the console
            # script installed next to the running interpreter. (pip installs
            # tau2 as <venv>/bin/tau2, but that bin/ is NOT on PATH unless the
            # venv is activated, so `which` misses it under `.venv/bin/python`.)
            sibling = Path(sys.executable).parent / "tau2"
            self.tau2_bin = Path(shutil.which("tau2") or (sibling if sibling.exists() else self.tau2_bin))
        work = Path(tempfile.mkdtemp(prefix=f"studio-tau2-{self.domain}-r{run_idx}-"))
        data_dir = self._build_data_dir(harness, work / "data")
        out_path = work / "results.json"
        cmd = self.build_cmd(task_ids, out_path, run_idx)
        env = os.environ.copy()
        env["TAU2_DATA_DIR"] = str(data_dir)
        # Inject the mutated agent instruction (read by patched tau2 source).
        if self.editable_instruction and harness.exists(AGENT_INSTRUCTION_FILE):
            env["TAU2_AGENT_INSTRUCTION"] = harness.read_file(AGENT_INSTRUCTION_FILE)
        log_path = work / "tau2.log"
        try:
            with open(log_path, "w") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0 or not out_path.exists():
                tail = log_path.read_text(errors="replace")[-2000:]
                raise BenchmarkExecutionError(
                    f"tau2 exited rc={proc.returncode} (results missing):\n{tail}"
                )
            scores = self._parse_results(out_path, task_ids, harness.content_hash())
            return scores
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _load_simulations(self, out_path: Path) -> list[dict]:
        """Read tau2 results as a list of plain dicts, format-agnostic.

        ``tau2 run --save-to X`` makes ``X`` a DIRECTORY holding a monolithic
        ``X/results.json`` with the simulations inline (NOT the dir format with
        a ``simulations/`` subdir — so ``Results.load`` is the wrong loader and
        returns nothing). We read the inline file directly, and only fall back
        to ``Results.load`` for the genuine simulations/-subdir variant."""
        p = Path(out_path)
        candidates = [p / "results.json", p] if p.is_dir() else [p]
        for c in candidates:
            try:
                if c.is_file():
                    sims = json.loads(c.read_text()).get("simulations")
                    if sims:
                        return sims
            except (ValueError, OSError):
                continue
        try:  # dir format with a simulations/ subdir
            from tau2.data_model.simulation import Results

            return [s.model_dump() for s in Results.load(p).simulations]
        except Exception:  # noqa: BLE001
            return []

    def _parse_results(self, out_path: Path, task_ids, harness_hash: str) -> dict[str, float]:
        sims = self._load_simulations(out_path)
        rewards: dict[str, list[float]] = {t: [] for t in task_ids}
        worst: dict[str, tuple[float, dict]] = {}  # task -> (reward, worst failing sim)
        for s in sims:
            tid = str(s.get("task_id"))
            if tid not in rewards:
                continue
            ri = s.get("reward_info") or {}
            r = ri.get("reward")
            rf = float(r) if isinstance(r, (int, float)) else 0.0
            if isinstance(r, (int, float)):
                rewards[tid].append(rf)
            # Keep the WORST failing trial's evidence (most informative), not
            # just the first — the localizer wants the clearest failure.
            if rf < 1.0 and (tid not in worst or rf < worst[tid][0]):
                worst[tid] = (rf, s)
        # per-task = mean reward over its trials (Pass^k-style aggregate in [0,1]).
        missing = [t for t in task_ids if not rewards[t]]
        if missing:
            raise BenchmarkExecutionError(
                f"tau2 produced no simulations for: {', '.join(missing[:5])}"
            )
        for _rf, sim in worst.values():
            self.evidence_store.put(harness_hash, self._build_evidence(sim))
        return {t: sum(v) / len(v) for t, v in rewards.items()}

    # Cap how much of a transcript we keep in memory for deep reads (tau2
    # dialogues are short; this is a guard against pathological runs).
    _MAX_FULL_MESSAGES = 60

    def _build_evidence(self, sim: dict) -> TaskEvidence:
        """Decode one simulation into structured evidence: which verifier checks
        failed (signals) + the causal transcript windows they point at. Never
        raises — a malformed ``reward_info`` degrades to a tail window."""
        tid = str(sim.get("task_id"))
        ri = sim.get("reward_info") or {}
        reward = float(ri.get("reward") or 0.0) if isinstance(ri.get("reward"), (int, float)) else 0.0
        trial = int(sim.get("trial") or 0)
        messages = sim.get("messages") if isinstance(sim.get("messages"), list) else []
        signals = _decode_signals(ri)
        windows = select_windows(messages, signals, task_id=tid, trial=trial)
        full = [_slim_message(m) for m in messages[: self._MAX_FULL_MESSAGES] if isinstance(m, dict)]
        return TaskEvidence(
            task_id=tid, reward=reward, trial=trial, signals=signals,
            windows=windows, transcript_len=len(messages), full_messages=full,
        )

    def last_trace(self, task_id: str, *, harness: Harness | None = None) -> str:
        if harness is None:
            return ""
        ev = self.evidence_store.get(harness.content_hash(), str(task_id))
        return to_flat_excerpt(ev) if ev else ""

    def last_evidence(self, task_id: str, *, harness: Harness | None = None):
        if harness is None:
            return None
        return self.evidence_store.get(harness.content_hash(), str(task_id))

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        required = list(self.policy_files)
        if self.editable_instruction:
            required.append(AGENT_INSTRUCTION_FILE)
        for f in required:
            if not harness.exists(f):
                return False, f"missing {f}"
            if not harness.read_file(f).strip():
                return False, f"{f} is empty"
        return True, ""

    def describe(self, task_id: str) -> str:
        return f"{self.domain}:{task_id}"


def tau2_part_map(domain: str = "airline", *, editable_instruction: bool | None = None) -> PartMap:
    """The editable surface: the domain policy file(s), and — once tau2's source
    can consume it — the agent instruction. Both are INSTRUCTIONS."""
    if editable_instruction is None:
        editable_instruction = instruction_injectable()
    files = list(policy_files_for(domain))
    if editable_instruction:
        files.append(AGENT_INSTRUCTION_FILE)
    return PartMap(parts={PartType.INSTRUCTIONS: files}, do_not_touch=[])


def tau2_seed_harness(
    domain: str, dest: Path, tau2_repo: Path = DEFAULT_TAU2_REPO,
    *, editable_instruction: bool | None = None,
) -> Harness:
    """Materialize the shipped baseline harness: the original domain policy
    file(s), plus tau2's default agent instruction when that lever is on."""
    if editable_instruction is None:
        editable_instruction = instruction_injectable(tau2_repo)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dom_dir = Path(tau2_repo) / "data" / "tau2" / "domains" / domain
    h = Harness(dest)
    for pf in policy_files_for(domain):
        h.write_file(pf, (dom_dir / pf).read_text())
    if editable_instruction:
        h.write_file(AGENT_INSTRUCTION_FILE, default_agent_instruction())
    return h
