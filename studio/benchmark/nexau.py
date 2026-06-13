"""NexAU benchmark adapter — score AHE's *exact* input harness on Terminal-Bench 2.

AHE (agentic-harness-engineering) optimizes the ``agents/code_agent_simple/``
NexAU agent and scores it with::

    harbor run --agent nexau --env docker --model gpt-5.4 \
        --ak config_path=<workspace>/code_agent.yaml --jobs-dir <dir> [-k K] \
        -p <task_dir> [--timeout-multiplier T]

This adapter makes harness_studio score the **same** harness on the **same**
harbor path, so a head-to-head is apples-to-apples: the only thing that differs
between the two systems is *the optimizer that produced the workspace*.

Design notes
------------
* The ``nexau`` harbor agent is registered only in AHE's pinned harbor (the git
  dep ``Curry09/harbor-LJH`` installed in AHE's ``.venv``). So we invoke
  ``<ahe_dir>/.venv/bin/harbor`` with ``cwd=ahe_dir`` and the venv on ``PATH``,
  exactly reproducing how ``evolve.py`` launches it.
* The actor model is ``gpt-5.4`` via the OpenAI Responses API. harbor reads the
  credentials from ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` in the
  subprocess env — we load those from AHE's ``.env`` (never logged).
* ``run_idx`` is threaded into a fresh ``--jobs-dir`` per call, so each repeated
  run is a genuinely independent rollout. Combined with the InstrumentedBenchmark
  cache key ``(hash, run_idx, task)`` this gives the gate *real* TB2 wobble
  instead of a cached constant.
* Task selection uses the locally-cached task directories
  (``~/.cache/harbor/tasks/<hash>/<task-name>/``). ``list_tasks`` enumerates a
  fixed subset; ``run`` symlinks the selected task dirs into a throwaway dataset
  dir and passes it as ``-p`` (one harbor invocation scores the whole batch with
  harbor-side concurrency).

The harbor-result parsing reuses :func:`studio.benchmark.kira.parse_harbor_results`
(a unit-tested pure function), so the scoring contract is shared and verified.
"""

from __future__ import annotations

import os
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..components.evidence import EvidenceStore, evidence_from_trace
from ..harness import Harness
from .base import Benchmark
from .kira import BenchmarkExecutionError, require_complete_harbor_results

# Portable: override on another machine with `export AHE_DIR=/path/to/agentic-harness-engineering`.
DEFAULT_AHE_DIR = Path(
    os.environ.get(
        "AHE_DIR",
        str(Path(__file__).resolve().parents[3] / "agentic-harness-engineering"),
    )
)
DEFAULT_TASK_CACHE = Path(os.environ.get("HARBOR_TASK_CACHE", str(Path.home() / ".cache" / "harbor" / "tasks")))
AGENT_CONFIG_FILENAME = "code_agent.yaml"


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` reader (``KEY=value``; strips quotes; ignores comments).

    Returns only the keys present; values are never printed by this module.
    """
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


# nexau api_type + credential resolution per provider, so the SAME nexau path
# can drive any backbone (gemini_rest for Gemini, openai_responses for GPT-5.x,
# anthropic_chat_completion for Claude). The actor's code_agent.yaml reads
# api_type from ${env.LLM_API_TYPE}.
_PROVIDER_API_TYPE = {
    "gemini": "gemini_rest",
    "openai": "openai_responses",
    "anthropic": "anthropic_chat_completion",
}
_PROVIDER_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}
# base_url env vars to consult PER PROVIDER — never let a global gemini
# LLM_BASE_URL leak into an OpenAI/Anthropic run.
_PROVIDER_BASE_ENV = {
    "gemini": ("GEMINI_BASE_URL", "LLM_BASE_URL", "GPT54_LLM_BASE_URL"),
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    "anthropic": ("ANTHROPIC_BASE_URL",),
}
_PROVIDER_BASE_URL = {"gemini": "https://generativelanguage.googleapis.com"}


def provider_of(model: str) -> str:
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gemini" in m or "google" in m:
        return "gemini"
    if any(x in m for x in ("gpt", "o1", "o3", "o4", "openai")):
        return "openai"
    return "openai"


def api_type_for(model: str, provider: str | None = None) -> str:
    prov = provider or provider_of(model)
    if prov not in _PROVIDER_API_TYPE:
        raise ValueError(f"unsupported NexAU provider {prov!r}")
    return _PROVIDER_API_TYPE[prov]


def load_llm_env(
    ahe_dir: Path, model: str, provider: str | None = None
) -> dict[str, str]:
    """Assemble the ``LLM_*`` env harbor's nexau agent needs, provider-aware.

    The actor's provider is inferred from ``model``; its key is read from the
    provider-native env var (e.g. ``GEMINI_API_KEY``/``OPENAI_API_KEY``), with the
    legacy ``LLM_API_KEY``/``GPT54_LLM_*`` (in env or AHE's ``.env``) as fallback.
    ``LLM_MODEL`` is always ``model`` and ``LLM_API_TYPE`` selects the nexau path.
    """
    env_file = parse_dotenv(Path(ahe_dir) / ".env")
    prov = provider or provider_of(model)
    if prov not in _PROVIDER_API_TYPE:
        raise ValueError(f"unsupported NexAU provider {prov!r}")

    def pick(*names: str) -> str:
        for n in names:
            v = os.environ.get(n) or env_file.get(n)
            if v:
                return v
        return ""

    out = {"LLM_MODEL": model, "LLM_API_TYPE": api_type_for(model, prov)}
    key = pick(*_PROVIDER_KEY_ENV.get(prov, ()), "LLM_API_KEY", "GPT54_LLM_API_KEY")
    base_url = pick(*_PROVIDER_BASE_ENV.get(prov, ())) or _PROVIDER_BASE_URL.get(prov, "")
    if key:
        out["LLM_API_KEY"] = key
        # also expose the provider-native var (nexau/litellm/openai_responses read it)
        for n in _PROVIDER_KEY_ENV.get(prov, ("OPENAI_API_KEY",)):
            out[n] = key
    if base_url:
        out["LLM_BASE_URL"] = base_url
    return out


class NexauBenchmark(Benchmark):
    def __init__(
        self,
        *,
        real: bool = False,
        ahe_dir: Path = DEFAULT_AHE_DIR,
        task_cache: Path = DEFAULT_TASK_CACHE,
        tasks: list[str] | None = None,
        model: str = "gemini-3.5-flash",
        provider: str | None = None,
        env: str = "docker",
        n_concurrent: int = 4,
        k: int = 1,
        timeout_multiplier: float = 3.0,
        force_build: bool = True,
        harbor_bin: Path | None = None,
        agent_config_filename: str = AGENT_CONFIG_FILENAME,
    ) -> None:
        self.real = real
        self.ahe_dir = Path(ahe_dir)
        self.task_cache = Path(task_cache)
        self.tasks = list(tasks or [])
        self.model = model
        self.provider = provider or provider_of(model)
        api_type_for(model, self.provider)  # fail fast on unsupported providers
        self.env = env
        self.n_concurrent = n_concurrent
        self.k = k
        self.timeout_multiplier = timeout_multiplier
        # Build the task image from its Dockerfile (arm64-native on this box; the
        # published docker_images can be x86-only) so the nexau runtime install
        # has apt + a working base.
        self.force_build = force_build
        self.agent_config_filename = agent_config_filename
        # Faithful to evolve.py: use AHE's pinned harbor (it registers `nexau`).
        self.harbor_bin = Path(harbor_bin) if harbor_bin else self.ahe_dir / ".venv" / "bin" / "harbor"
        # harness_hash -> {task_id -> failure excerpt} from that harness's most
        # recent run. Versioning by harness keeps a candidate's gate run from
        # being attributed to the live harness (the diagnoser reads these).
        # Stored in-memory so the on-disk jobs dir can be deleted right after
        # each eval; only the most recent few harness versions are retained.
        self._traces: dict[str, dict[str, str]] = {}
        # Structured evidence (for the localizer + editor); populated alongside
        # the flat _traces. last_trace stays back-compatible.
        self.evidence_store = EvidenceStore()

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

    def build_cmd(
        self, harness: Harness, task_ids: list[str], jobs_dir: Path, dataset_dir: Path,
        *, force_build: bool | None = None,
    ) -> list[str]:
        config_path = (harness.root / self.agent_config_filename).resolve()
        cmd = [
            str(self.harbor_bin), "run",
            "--agent", "nexau",
            "--env", self.env,
            "--model", self.model,
            "--n-concurrent", str(self.n_concurrent),
            "--ak", f"config_path={config_path}",
            "--jobs-dir", str(jobs_dir),
            "-p", str(dataset_dir),
        ]
        if self.force_build if force_build is None else force_build:
            cmd += ["--force-build"]
        if self.k > 1:
            cmd += ["-k", str(self.k)]
        if self.timeout_multiplier and self.timeout_multiplier != 1.0:
            cmd += ["--timeout-multiplier", str(self.timeout_multiplier)]
        return cmd

    def _subprocess_env(self) -> dict[str, str]:
        sub_env = os.environ.copy()
        sub_env.update(load_llm_env(self.ahe_dir, self.model, self.provider))
        # Docker path: nexau is installed into the task container at runtime
        # (the pre-baked /opt/nexau-venv is E2B-only). Honor an explicit override.
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
                "Docker + AHE's harbor + gpt-5.4 credentials"
            )
        task_ids = list(task_ids)
        if not task_ids:
            return {}
        if not self.harbor_bin.exists():
            raise RuntimeError(
                f"harbor not found at {self.harbor_bin}; expected AHE's pinned harbor "
                f"(run `uv sync` in {self.ahe_dir})"
            )
        work = Path(tempfile.mkdtemp(prefix=f"studio-nexau-r{run_idx}-"))
        dataset_dir = work / "dataset"
        jobs_dir = work / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        self._link_dataset(task_ids, dataset_dir)
        # ALWAYS honor force_build per invocation: without --force-build harbor
        # switches to its prebuilt-image compose path, which is x86-only and
        # crashes agent setup on arm64 hosts. Docker layer caching keeps the
        # repeat "builds" cheap; skipping the flag does not.
        cmd = self.build_cmd(harness, task_ids, jobs_dir, dataset_dir)
        log_path = work / "harbor.log"
        try:
            with open(log_path, "w") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.run(
                    cmd, cwd=str(self.ahe_dir), env=self._subprocess_env(),
                    stdout=log, stderr=subprocess.STDOUT,
                )
            self._capture_traces(jobs_dir, task_ids, harness.content_hash())
            if proc.returncode != 0:
                tail = log_path.read_text(errors="replace")[-2000:]
                raise BenchmarkExecutionError(
                    f"Harbor exited with rc={proc.returncode}:\n{tail}"
                )
            return require_complete_harbor_results(
                jobs_dir, task_ids, expected_trials=max(1, self.k)
            )
        finally:
            # Free disk immediately: the excerpts we need are now in memory, so
            # per-trial logs and traces do not accumulate across a run.
            shutil.rmtree(work, ignore_errors=True)

    # --- trace-feeding: surface why a task failed (PRD §5.1 trajectory) ---

    _TRACE_HASHES = 8  # harness versions to retain traces for

    def _capture_traces(self, jobs_dir: Path, task_ids: list[str], harness_hash: str) -> None:
        """Extract a concise failure excerpt per task into memory (so the jobs
        dir can be deleted right after), keyed by the harness that produced it."""
        bucket = self._traces.pop(harness_hash, {})
        self._traces[harness_hash] = bucket  # re-insert as most recent
        for reward_file in Path(jobs_dir).rglob("verifier/reward.txt"):
            trial_dir = reward_file.parent.parent  # <task>__<trial>/
            tid = trial_dir.name.split("__", 1)[0]
            if tid in task_ids:
                bucket[tid] = self._extract_excerpt(trial_dir)
                self.evidence_store.put(harness_hash, self._trial_evidence(trial_dir, tid))
        while len(self._traces) > self._TRACE_HASHES:
            self._traces.pop(next(iter(self._traces)))

    @staticmethod
    def _trial_evidence(trial_dir: Path, tid: str):
        """Structured evidence for one trial: the test verifier output as a
        signal + the agent trajectory messages as the causal window."""
        reward = 0.0
        rf = trial_dir / "verifier" / "reward.txt"
        try:
            reward = float(rf.read_text().strip()) if rf.exists() else 0.0
        except (OSError, ValueError):
            reward = 0.0
        verifier = trial_dir / "verifier" / "test-stdout.txt"
        vtext = verifier.read_text(errors="replace") if verifier.exists() else ""
        msgs = NexauBenchmark._tracer_messages(
            trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json")
        return evidence_from_trace(tid, reward, verifier_output=vtext, messages=msgs)

    @staticmethod
    def _tracer_messages(tracer: Path) -> list[dict]:
        if not tracer.exists():
            return []
        try:
            import json
            data = json.loads(tracer.read_text(errors="replace"))
        except (OSError, ValueError):
            return []
        msgs = data.get("messages") if isinstance(data, dict) else None
        return [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else []

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
        # 2. The agent's last actions / final output.
        excerpt = NexauBenchmark._tracer_excerpt(trial_dir / "agent" / "nexau_in_memory_tracer.cleaned.json")
        if excerpt:
            parts.append("agent trajectory (tail):\n" + excerpt)
        return "\n\n".join(parts)[:2400]

    def last_trace(self, task_id: str, *, harness: Harness | None = None) -> str:
        if harness is None:
            return ""  # without a harness identity we'd risk returning the wrong run's trace
        return self._traces.get(harness.content_hash(), {}).get(task_id, "")

    def last_evidence(self, task_id: str, *, harness: Harness | None = None):
        if harness is None:
            return None
        return self.evidence_store.get(harness.content_hash(), task_id)

    @staticmethod
    def _tracer_excerpt(tracer: Path) -> str:
        if not tracer.exists():
            return ""
        try:
            import json

            data = json.loads(tracer.read_text(errors="replace"))
        except (OSError, ValueError):
            return ""
        msgs = data.get("messages") if isinstance(data, dict) else None
        if isinstance(msgs, list) and msgs:
            chunks: list[str] = []
            for m in msgs[-4:]:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "?")
                content = m.get("content")
                if isinstance(content, list):  # tool/content blocks
                    content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
                chunks.append(f"[{role}] {str(content)[:600]}")
            return "\n".join(chunks)[-1400:]
        out = data.get("output") if isinstance(data, dict) else None
        return str(out)[-1400:] if out else ""

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
