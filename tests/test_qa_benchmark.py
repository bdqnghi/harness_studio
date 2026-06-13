"""The generic single-turn QA adapter + the GSM8K grader/suite wiring."""

from pathlib import Path

from studio.benchmark.qa import QABenchmark, QATask
from studio.benchmark.qa_suites import (
    _grade_gsm8k, _grade_hotpot, _hotpot_context, get_suite,
)
from studio.harness import Harness


def _harness(tmp_path, prompt="answer the question") -> Harness:
    h = Harness(tmp_path / "h")
    h.write_file("system_prompt.md", prompt)
    return h


# --- GSM8K grader (deterministic, no network) ---

def _t(gold):
    return QATask(id="0", question="q", gold=[gold])


def test_gsm8k_grader_marker_and_last_int_and_tag():
    assert _grade_gsm8k("blah\n#### 42", _t("42")) == 1.0
    assert _grade_gsm8k("the answer is 42", _t("42")) == 1.0          # last int fallback
    assert _grade_gsm8k("<answer>42</answer>", _t("42")) == 1.0       # tag
    assert _grade_gsm8k("I think it is 7", _t("42")) == 0.0
    assert _grade_gsm8k("no number here", _t("42")) == 0.0
    assert _grade_gsm8k("#### 1,234", _t("1234")) == 1.0              # comma-stripped


# --- QABenchmark mechanics (model stubbed; no real calls) ---

class _Bench(QABenchmark):
    """QABenchmark with the LLM call replaced by a scripted answer map."""

    def __init__(self, answers, **kw):
        super().__init__(**kw)
        self._answers = answers

    def _answer_once(self, system, task):
        return self._answers.get(task.id, "")


def _grade_exact(out, task):
    return 1.0 if out.strip() == task.gold[0] else 0.0


def test_run_scores_and_records_failure_trace(tmp_path):
    tasks = [QATask(id="a", question="qa", gold=["yes"]),
             QATask(id="b", question="qb", gold=["yes"])]
    b = _Bench({"a": "yes", "b": "no"}, tasks=tasks, grader=_grade_exact,
               model="stub", k=1, n_concurrent=2)
    h = _harness(tmp_path)
    scores = b.run(h, ["a", "b"], run_idx=0)
    assert scores == {"a": 1.0, "b": 0.0}
    # the failing task left a trace versioned by harness hash; the passing one didn't
    assert "model said: no" in b.last_trace("b", harness=h)
    assert b.last_trace("a", harness=h) == ""


def test_boot_check_requires_nonempty_prompt(tmp_path):
    b = _Bench({}, tasks=[QATask(id="a", question="q")], grader=_grade_exact, model="stub")
    good = _harness(tmp_path)
    assert b.boot_check(good)[0] is True
    empty = Harness(tmp_path / "e"); empty.write_file("system_prompt.md", "  ")
    assert b.boot_check(empty)[0] is False
    missing = Harness(tmp_path / "m"); missing.write_file("other.md", "x")
    assert b.boot_check(missing)[0] is False


def test_k_rollouts_average(tmp_path):
    # k=2; a flaky answer (right once, wrong once) -> 0.5
    class _Flaky(QABenchmark):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._n = 0

        def _answer_once(self, system, task):
            self._n += 1
            return "yes" if self._n % 2 == 1 else "no"

    b = _Flaky(tasks=[QATask(id="a", question="q", gold=["yes"])],
               grader=_grade_exact, model="stub", k=2, n_concurrent=1)
    assert b.run(_harness(tmp_path), ["a"], run_idx=0)["a"] == 0.5


# --- suite registration ---

def test_gsm8k_suite_registered_and_seed_prompt():
    s = get_suite("gsm8k")
    assert s.name == "gsm8k"
    assert "####" in s.seed_prompt  # seed instructs the parseable format


# --- HotpotQA grader + context rendering (deterministic, no network) ---

def test_hotpot_f1_grader():
    t = QATask(id="0", question="q", gold=["Scott Derrickson"])
    assert _grade_hotpot("<answer>Scott Derrickson</answer>", t) == 1.0   # exact -> F1 1.0
    assert _grade_hotpot("<answer>no idea</answer>", t) == 0.0            # disjoint -> 0
    partial = _grade_hotpot("<answer>Scott</answer>", t)
    assert 0.0 < partial < 1.0                                           # token overlap


def test_hotpot_context_renders_numbered_sources():
    ctx = {"title": ["A", "B"], "sentences": [["s1.", "s2."], ["t1."]]}
    rendered = _hotpot_context(ctx)
    assert "[1] A: s1.s2." in rendered and "[2] B: t1." in rendered


def test_hotpot_suite_registered():
    s = get_suite("hotpot")
    assert s.name == "hotpot" and s.default_limit == 300
