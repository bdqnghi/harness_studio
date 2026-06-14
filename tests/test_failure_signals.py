"""FailureSignal extraction + the structured diagnose path (deterministic spine)."""

from studio.core.evidence import TaskEvidence, VerifierSignal
from studio.stages.optimize.diagnose import diagnoser, signals
from studio.stages.optimize.diagnose.signals import FailureSignal


def _ev(task_id, reward, sigs):
    return TaskEvidence(task_id=task_id, reward=reward,
                        signals=[VerifierSignal(*s) for s in sigs])


# --- from_evidence (deterministic) ---

def test_from_evidence_extracts_failed_checks_and_channels():
    ev = _ev("t1", 0.0, [
        ("action", "modify_order", False, "expected action modify_order({...})"),
        ("db", "", False, "final database state does not match the target"),
        ("communicate", "", True, ""),   # passed -> not a failure
    ])
    fs = signals.from_evidence(ev, score=0.0)
    assert fs.task_id == "t1"
    assert len(fs.failed) == 2                       # only the failed checks
    assert fs.failed_channels == ["action", "db"]
    assert any("modify_order" in d for d in fs.gt_diff)
    assert fs.consistent is True                     # score 0.0 <= 0.2


def test_flaky_task_marked_inconsistent():
    ev = _ev("t2", 0.0, [("db", "", False, "db mismatch")])
    assert signals.from_evidence(ev, score=0.5).consistent is False   # mixed -> flaky
    assert signals.from_evidence(ev, score=0.0).consistent is True


def test_signature_groups_same_failed_checks():
    a = signals.from_evidence(_ev("a", 0.0, [("db", "", False, "x")]))
    b = signals.from_evidence(_ev("b", 0.0, [("db", "", False, "y")]))   # same check, diff detail
    c = signals.from_evidence(_ev("c", 0.0, [("action", "cancel", False, "z")]))
    assert signals.signature(a) == signals.signature(b)
    assert signals.signature(a) != signals.signature(c)


def test_empty_failed_set_degrades_to_other():
    fs = signals.from_evidence(_ev("t", 0.0, []))
    assert signals.signature(fs) == frozenset({("other", "")})


# --- structured diagnose: deterministic membership + counts, LLM only names ---

class _StubBackend:
    def __init__(self, names): self._names = names; self.calls = 0
    def prompt_json(self, prompt, schema, *, tag="", model=None):
        self.calls += 1
        return self._names


def _recs():
    return [
        signals.from_evidence(_ev("t1", 0.0, [("db", "", False, "db mismatch")])),
        signals.from_evidence(_ev("t2", 0.0, [("db", "", False, "db mismatch")])),
        signals.from_evidence(_ev("t3", 0.0, [("action", "cancel", False, "missing cancel")])),
    ]


def test_structured_diagnose_counts_are_deterministic():
    be = _StubBackend([
        {"group_id": "g0", "root_cause": "skips db write", "blamed_part": "instructions"},
        {"group_id": "g1", "root_cause": "missing cancel", "blamed_part": "instructions"},
    ])
    out = diagnoser.diagnose(be, [], records=_recs())
    by_tasks = {tuple(sorted(p["failing_task_ids"])): p for p in out}
    assert ("t1", "t2") in by_tasks and ("t3",) in by_tasks         # grouped by signature
    assert by_tasks[("t1", "t2")]["tasks_affected"] == 2            # deterministic count
    assert by_tasks[("t1", "t2")]["root_cause"] == "skips db write" # LLM naming applied
    assert "db" in by_tasks[("t1", "t2")]["verifier_cause"]         # ground-truth signature


def test_structured_diagnose_survives_llm_failure():
    class _Boom:
        def prompt_json(self, *a, **k): raise RuntimeError("llm down")
    out = diagnoser.diagnose(_Boom(), [], records=_recs())
    assert len(out) == 2                                            # still grouped + counted
    assert all(p["tasks_affected"] >= 1 for p in out)
    assert all(p["verifier_cause"] for p in out)                   # ground truth present


def test_legacy_path_when_no_records():
    from studio.stages.optimize.diagnose.runner import Failure
    be = _StubBackend([{"pattern_id": "p1", "root_cause": "x",
                        "failing_task_ids": ["t1"], "blamed_part": "instructions"}])
    out = diagnoser.diagnose(be, [Failure("t1", "desc", trace="boom")])
    assert out and out[0]["pattern_id"] == "p1"
    assert out[0]["tasks_affected"] == 1                           # default-filled
