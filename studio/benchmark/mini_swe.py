"""mini-swe-agent benchmark adapter — score a mutated mini-swe-agent harness.

This mirrors :class:`studio.benchmark.nexau.NexauBenchmark` closely so the two
targets share the *same* harbor path and the *same* scoring contract — the only
thing that differs is which agent harbor runs and which harness files the
optimizer is allowed to mutate.

harbor already registers ``--agent mini-swe-agent`` and runs it as::

    mini -m <provider/model> -t <task> -y -o <traj> -l 0 --exit-immediately

Per-task reward lands at ``<task>__<trial>/verifier/reward.txt`` exactly like
nexau, so :func:`parse_harbor_results` works unchanged.

Differences vs nexau
--------------------
* The harbor agent is ``mini-swe-agent`` (not ``nexau``).
* The stock harbor command takes *no* config dir, so ``build_cmd`` does not pass
  ``--ak config_path=...``.
* mini-swe-agent uses litellm natively, so the actor model string is in
  litellm format (e.g. ``gemini/gemini-3.5-flash`` or ``gpt-5.4``).
* The mutated harness is advertised to harbor via the ``MSWEA_HARNESS_DIR`` env
  var (a forthcoming harbor patch mounts + editable-installs that dir and sets
  ``MSWEA_MINI_CONFIG_PATH``); the benchmark just points at where it lives.
"""

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..harness import Harness
from ..parts import PartMap, PartType
from .base import Benchmark
# Reuse nexau's unit-tested helpers so the scoring/credential contract is shared.
from .nexau import DEFAULT_AHE_DIR, DEFAULT_TASK_CACHE, load_llm_env, parse_harbor_results

# Default actor in litellm format (mini-swe-agent uses litellm natively).
DEFAULT_MODEL = "gemini/gemini-3.5-flash"


class MiniSweBenchmark(Benchmark):
    def __init__(
        self,
        *,
        real: bool = False,
        ahe_dir: Path = DEFAULT_AHE_DIR,
        task_cache: Path = DEFAULT_TASK_CACHE,
        tasks: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        env: str = "docker",
        n_concurrent: int = 4,
        k: int = 1,
        timeout_multiplier: float = 3.0,
        force_build: bool = True,
        harbor_bin: Path | None = None,
    ) -> None:
        self.real = real
        self.ahe_dir = Path(ahe_dir)
        self.task_cache = Path(task_cache)
        self.tasks = list(tasks or [])
        self.model = model
        self.env = env
        self.n_concurrent = n_concurrent
        self.k = k
        self.timeout_multiplier = timeout_multiplier
        # Build the task image from its Dockerfile so the runtime install has a
        # working base (the published docker_images can be x86-only).
        self.force_build = force_build
        # Reuse AHE's pinned harbor (it registers the `mini-swe-agent` agent).
        self.harbor_bin = Path(harbor_bin) if harbor_bin else self.ahe_dir / ".venv" / "bin" / "harbor"
        # task_id -> failure excerpt from the most recent run (for trace-feeding).
        self._traces: dict[str, str] = {}

    # --- task discovery ---

    def _task_dir(self, task_id: str) -> Path:
        """Resolve a task name to its cached task directory.

        Cache layout is ``<cache>/<content-hash>/<task-name>/``; a task name is
        unique, so we glob for it.
        """
        matches = sorted(self.task_cache.glob(f"*/{task_id}"))
        matches = [m for m in matches if m.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"task {task_id!r} not found under {self.task_cache} "
                f"(download it with `harbor datasets download terminal-bench-sample@2.0`)"
            )
        return matches[0]

    def list_tasks(self) -> list[str]:
        if self.tasks:
            return list(self.tasks)
        if not self.real:
            return []
        # Enumerate every cached task name (the leaf dir under each hash dir).
        names: list[str] = []
        for hash_dir in sorted(self.task_cache.iterdir()):
            if not hash_dir.is_dir():
                continue
            for task_dir in sorted(hash_dir.iterdir()):
                if task_dir.is_dir() and (task_dir / "task.toml").exists():
                    names.append(task_dir.name)
        return names

    # --- command construction (pure; used by dry-runs and run) ---

    def build_cmd(self, harness: Harness, task_ids: list[str], jobs_dir: Path, dataset_dir: Path) -> list[str]:
        # Stock harbor mini-swe-agent command takes no config dir, so (unlike
        # nexau) there is no `--ak config_path=...`.
        cmd = [
            str(self.harbor_bin), "run",
            "--agent", "mini-swe-agent",
            "--env", self.env,
            "--model", self.model,
            "--n-concurrent", str(self.n_concurrent),
            "--jobs-dir", str(jobs_dir),
            "-p", str(dataset_dir),
        ]
        if self.force_build:
            cmd += ["--force-build"]
        if self.k > 1:
            cmd += ["-k", str(self.k)]
        if self.timeout_multiplier and self.timeout_multiplier != 1.0:
            cmd += ["--timeout-multiplier", str(self.timeout_multiplier)]
        return cmd

    def _subprocess_env(self, harness: Harness) -> dict[str, str]:
        sub_env = os.environ.copy()
        sub_env.update(load_llm_env(self.ahe_dir, self.model))
        # harness injection (harbor patch honors MSWEA_HARNESS_DIR): advertise
        # where the mutated mini-swe-agent harness lives; the patched harbor
        # mounts + editable-installs it and sets MSWEA_MINI_CONFIG_PATH.
        sub_env["MSWEA_HARNESS_DIR"] = str(harness.root)
        sub_env.setdefault("USE_BP_E2B", "False")
        venv_bin = str(self.ahe_dir / ".venv" / "bin")
        sub_env["PATH"] = venv_bin + os.pathsep + sub_env.get("PATH", "")
        return sub_env

    def _link_dataset(self, task_ids: list[str], dest: Path) -> None:
        """Symlink the selected cached task dirs into a throwaway dataset dir."""
        dest.mkdir(parents=True, exist_ok=True)
        for t in task_ids:
            src = self._task_dir(t)
            link = dest / t
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src, target_is_directory=True)

    # --- scoring ---

    def run(self, harness: Harness, task_ids, *, run_idx: int = 0) -> dict[str, float]:
        if not self.real:
            raise NotImplementedError(
                "real Terminal-Bench scoring is opt-in (real=True) and needs "
                "Docker + AHE's harbor + actor model credentials"
            )
        task_ids = list(task_ids)
        if not task_ids:
            return {}
        if not self.harbor_bin.exists():
            raise RuntimeError(
                f"harbor not found at {self.harbor_bin}; expected AHE's pinned harbor "
                f"(run `uv sync` in {self.ahe_dir})"
            )
        work = Path(tempfile.mkdtemp(prefix=f"studio-mini-swe-r{run_idx}-"))
        dataset_dir = work / "dataset"
        jobs_dir = work / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self._link_dataset(task_ids, dataset_dir)
        cmd = self.build_cmd(harness, task_ids, jobs_dir, dataset_dir)
        log_path = work / "harbor.log"
        with open(log_path, "w") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            # Do not raise on non-zero exit: a partial run still produces reward
            # files for the tasks that finished; parse_harbor_results scores a
            # missing task as 0.0 (a failed run, not absent data).
            subprocess.run(
                cmd, cwd=str(self.ahe_dir), env=self._subprocess_env(harness),
                stdout=log, stderr=subprocess.STDOUT,
            )
        results = parse_harbor_results(jobs_dir, task_ids)
        self._capture_traces(jobs_dir, task_ids)
        # Free disk immediately: the excerpts we need are now in memory.
        shutil.rmtree(work, ignore_errors=True)
        return results

    # --- trace-feeding: surface why a task failed (PRD §5.1 trajectory) ---

    def _capture_traces(self, jobs_dir: Path, task_ids: list[str]) -> None:
        """Extract a concise failure excerpt per task into memory (so the jobs
        dir can be deleted right after)."""
        for reward_file in Path(jobs_dir).rglob("verifier/reward.txt"):
            trial_dir = reward_file.parent.parent  # <task>__<trial>/
            tid = trial_dir.name.split("__", 1)[0]
            if tid in task_ids:
                self._traces[tid] = self._extract_excerpt(trial_dir)

    @staticmethod
    def _extract_excerpt(trial_dir: Path) -> str:
        parts: list[str] = []
        # 1. Why the verifier failed (the most informative signal).
        verifier = trial_dir / "verifier" / "test-stdout.txt"
        if verifier.exists():
            try:
                txt = verifier.read_text(errors="replace").strip()
                if txt:
                    parts.append("verifier output (tail):\n" + txt[-1000:])
            except OSError:
                pass
        # 2. The agent's last actions / final output (mini-swe-agent trajectory).
        excerpt = MiniSweBenchmark._traj_excerpt(trial_dir / "agent")
        if excerpt:
            parts.append("agent trajectory (tail):\n" + excerpt)
        return "\n\n".join(parts)[:2400]

    def last_trace(self, task_id: str) -> str:
        return self._traces.get(task_id, "")

    @staticmethod
    def _traj_excerpt(agent_dir: Path) -> str:
        """Tail of the mini-swe-agent trajectory (``-o <traj>`` JSON output)."""
        if not agent_dir.is_dir():
            return ""
        trajs = sorted(agent_dir.glob("*.json")) + sorted(agent_dir.glob("*.traj"))
        if not trajs:
            return ""
        try:
            import json

            data = json.loads(trajs[0].read_text(errors="replace"))
        except (OSError, ValueError):
            return ""
        msgs = None
        if isinstance(data, dict):
            msgs = data.get("messages") or data.get("history") or data.get("trajectory")
        elif isinstance(data, list):
            msgs = data
        if isinstance(msgs, list) and msgs:
            chunks: list[str] = []
            for m in msgs[-4:]:
                if not isinstance(m, dict):
                    chunks.append(str(m)[:600])
                    continue
                role = m.get("role", "?")
                content = m.get("content")
                if isinstance(content, list):  # tool/content blocks
                    content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
                chunks.append(f"[{role}] {str(content)[:600]}")
            return "\n".join(chunks)[-1400:]
        return str(data)[-1400:]

    # --- free structural pre-gate ---

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as tmp:
            for rel in harness.files():
                path = harness.root / rel
                if rel.endswith(".py"):
                    try:
                        py_compile.compile(str(path), cfile=str(Path(tmp) / "out.pyc"), doraise=True)
                    except py_compile.PyCompileError as e:
                        return False, f"{rel}: {e.msg}"
                elif rel.endswith((".yaml", ".yml")):
                    try:
                        import yaml  # optional; skip the check if unavailable

                        yaml.safe_load(path.read_text())
                    except ModuleNotFoundError:
                        pass
                    except Exception as e:  # noqa: BLE001 - any parse error is a boot failure
                        return False, f"{rel}: invalid YAML: {e}"
        return True, ""

    def describe(self, task_id: str) -> str:
        try:
            td = self._task_dir(task_id)
            inst = td / "instruction.md"
            if inst.exists():
                return f"{task_id}: {inst.read_text()[:200].strip()}"
        except FileNotFoundError:
            pass
        return task_id


def mini_swe_part_map() -> PartMap:
    """Explicit, fair search space for the mini-swe-agent harness.

    Directory entries (trailing ``/``) let the optimizer ADD new config presets,
    tools, and environments (not just edit existing files), mirroring the freedom
    AHE has over the NexAU harness. Inert infra (``pyproject.toml``, the package
    ``__init__``, and the ``run/`` CLI entrypoints) is off-limits.

    Paths match the installed package layout (``src/minisweagent/``):
    INSTRUCTIONS  -> config YAMLs (system_template/instance_template),
    TOOL_*        -> the action parsers + environments that define/run tools,
    MIDDLEWARE    -> the litellm model wrapper (observation/format/model_kwargs),
    MEMORY        -> the default agent loop (history management).
    SKILLS/SUBAGENTS are absent in this harness.
    """
    return PartMap(
        parts={
            PartType.INSTRUCTIONS: ["src/minisweagent/config/"],
            PartType.TOOL_DESCRIPTIONS: ["src/minisweagent/models/utils/", "src/minisweagent/environments/"],
            PartType.TOOL_CODE: ["src/minisweagent/models/utils/", "src/minisweagent/environments/"],
            PartType.MIDDLEWARE: ["src/minisweagent/models/litellm_model.py"],
            PartType.MEMORY: ["src/minisweagent/agents/default.py"],
        },
        do_not_touch=["pyproject.toml", "src/minisweagent/__init__.py", "src/minisweagent/run/"],
    )
