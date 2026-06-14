"""A generic single-turn QA benchmark adapter — the docker-free spine.

Many benchmarks from the agent-optimization literature (GSM8K, IFEval,
HotpotQA-distractor, SearchQA, GPQA, BBH, …) share one shape: present a
question (optionally with provided context), let the model answer in one turn,
then grade the answer with a *deterministic, self-contained* grader. No docker,
no live web, no tools. That makes them ideal early-signal targets for SHO.

THE EDITABLE HARNESS here is the agent's **prompt policy** — by default a single
``system_prompt.md`` (the instructions the model follows + the answer format it
must emit). The optimizer hill-climbs that prompt; the grader is the fixed trust
anchor. A new benchmark = a list of :class:`QATask` + a ``grader`` callable
(see ``qa_suites.py``); everything else is shared.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..core.evidence import EvidenceStore, TaskEvidence, VerifierSignal
from ..core.harness import Harness
from .base import Benchmark

PROMPT_FILE = "system_prompt.md"


@dataclass
class QATask:
    """One single-turn QA item. ``gold`` holds acceptable answers (for graders
    that need them); ``meta`` carries grader-specific data (e.g. IFEval's
    instruction constraints). ``context`` is provided reading material, if any
    (e.g. HotpotQA's inline paragraphs) — NOT fetched live."""

    id: str
    question: str
    gold: list[str] = field(default_factory=list)
    context: str = ""
    meta: dict = field(default_factory=dict)


# A grader maps (model_output_text, task) -> score in [0, 1]. Pure, deterministic.
Grader = Callable[[str, QATask], float]


class QABenchmark(Benchmark):
    """Scores a prompt-policy harness on single-turn QA tasks via an LLM call."""

    def __init__(
        self,
        *,
        tasks: list[QATask],
        grader: Grader,
        model: str,
        k: int = 1,
        n_concurrent: int = 8,
        real: bool = True,
        prompt_file: str = PROMPT_FILE,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        self._tasks = {t.id: t for t in tasks}
        self._order = [t.id for t in tasks]
        self.grader = grader
        self.model = model
        self.k = max(1, k)
        self.n_concurrent = max(1, n_concurrent)
        self.real = real
        self.prompt_file = prompt_file
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Per-(harness_hash, task_id) failure excerpt for the Diagnoser. Versioned
        # by harness so a candidate's run is never attributed to the live harness.
        self._traces: dict[tuple[str, str], str] = {}
        # Structured evidence (one "answer" VerifierSignal per failing task) so the
        # diagnosis engine gets the same grounded path as tau2 (gold vs prediction).
        self.evidence_store = EvidenceStore()

    # --- discovery ---

    def list_tasks(self) -> list[str]:
        return list(self._order)

    def describe(self, task_id: str) -> str:
        t = self._tasks.get(task_id)
        if t is None:
            return task_id
        q = t.question.strip().replace("\n", " ")
        return q[:200] + ("…" if len(q) > 200 else "")

    def boot_check(self, harness: Harness) -> tuple[bool, str]:
        if not harness.exists(self.prompt_file):
            return False, f"harness is missing {self.prompt_file}"
        if not harness.read_file(self.prompt_file).strip():
            return False, f"{self.prompt_file} is empty"
        return True, ""

    def last_trace(self, task_id: str, *, harness: Harness | None = None) -> str:
        if harness is not None:
            return self._traces.get((harness.content_hash(), task_id), "")
        # Newest excerpt for this task across harnesses (best-effort).
        for (h, t), v in reversed(list(self._traces.items())):
            if t == task_id:
                return v
        return ""

    def last_evidence(self, task_id: str, *, harness: Harness | None = None):
        """Structured evidence (the gold-vs-prediction check) for the diagnosis
        engine; None when unavailable (e.g. a passing task or no harness scope)."""
        if harness is None:
            return None
        return self.evidence_store.get(harness.content_hash(), str(task_id))

    # --- scoring ---

    def _user_content(self, task: QATask) -> str:
        if task.context:
            return f"{task.context}\n\n{task.question}"
        return task.question

    def _answer_once(self, system: str, task: QATask) -> str:
        """One LLM completion; returns the raw text (empty string on error)."""
        if not self.real:
            return ""
        import litellm

        try:
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": self._user_content(task)},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                drop_params=True,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 — one bad call must not kill the batch
            return f"__ERROR__ {type(exc).__name__}: {exc}"

    def _score_task(self, system: str, task: QATask, harness_hash: str) -> float:
        """Mean grader score over k rollouts; records a trace on (any) failure."""
        scores: list[float] = []
        worst_output = ""
        worst = 1.0
        for _ in range(self.k):
            out = self._answer_once(system, task)
            s = 0.0 if out.startswith("__ERROR__") else self.grader(out, task)
            scores.append(s)
            if s <= worst:
                worst, worst_output = s, out
        mean = sum(scores) / len(scores) if scores else 0.0
        if worst < 1.0:
            gold = " | ".join(task.gold) if task.gold else "(grader-defined)"
            said = worst_output.strip()[:600]
            self._traces[(harness_hash, task.id)] = (
                f"Q: {self.describe(task.id)}\n"
                f"gold: {gold}\n"
                f"model said: {said}\n"
                f"score: {worst:.2f}"
            )
            # Structured evidence: one "answer" check that failed (the gold diff).
            self.evidence_store.put(harness_hash, TaskEvidence(
                task_id=task.id, reward=worst,
                signals=[VerifierSignal(
                    kind="answer", name="match", passed=False,
                    detail=f"answered '{said}', expected one of [{gold}] (score {worst:.2f})")],
            ))
        return mean

    def run(
        self, harness: Harness, task_ids: list[str], *, run_idx: int = 0
    ) -> dict[str, float]:
        system = harness.read_file(self.prompt_file)
        hh = harness.content_hash()
        ids = [t for t in task_ids if t in self._tasks]

        def work(tid: str) -> tuple[str, float]:
            return tid, self._score_task(system, self._tasks[tid], hh)

        out: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=self.n_concurrent) as pool:
            for tid, score in pool.map(work, ids):
                out[tid] = score
        return out
