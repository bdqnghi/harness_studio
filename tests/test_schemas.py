import pytest

from studio import schemas


def test_valid_review_passes():
    schemas.validate({"keep": ["a"], "drop": [{"strategy_id": "b", "reason": "x"}]},
                     schemas.REVIEW)


def test_missing_required_key_fails():
    with pytest.raises(schemas.SchemaError):
        schemas.validate({"keep": ["a"]}, schemas.REVIEW)


def test_additional_property_fails():
    with pytest.raises(schemas.SchemaError):
        schemas.validate({"order": [], "extra": 1}, schemas.RANKING)


def test_type_mismatch_fails():
    with pytest.raises(schemas.SchemaError):
        schemas.validate({"order": "not-a-list"}, schemas.RANKING)


def test_bool_is_not_integer():
    with pytest.raises(schemas.SchemaError):
        schemas.validate(True, {"type": "integer"})


def test_diagnosis_array():
    data = [{
        "pattern_id": "p1", "root_cause": "timeout",
        "failing_task_ids": ["t1"], "blamed_part": "tool_code", "confidence": 0.8,
    }]
    schemas.validate(data, schemas.DIAGNOSIS)


def test_diagnosis_signature_fields_validate():
    from studio import schemas

    full = [{
        "pattern_id": "p1", "root_cause": "r", "failing_task_ids": ["t"],
        "blamed_part": "tool_code", "verifier_cause": "assertion failed",
        "agent_mechanism": "never ran the test", "addressable": True,
    }]
    schemas.validate(full, schemas.DIAGNOSIS)  # signature triple accepted
    minimal = [{"pattern_id": "p1", "root_cause": "r",
                "failing_task_ids": ["t"], "blamed_part": "tool_code"}]
    schemas.validate(minimal, schemas.DIAGNOSIS)  # still optional
    try:
        schemas.validate(
            [{**minimal[0], "addressable": "yes"}], schemas.DIAGNOSIS)
        raise AssertionError("non-boolean addressable should fail")
    except schemas.SchemaError:
        pass


def test_diagnose_default_fills_signature():
    from studio.stages.optimize.diagnose import diagnoser, runner

    class _Backend:
        def prompt_json(self, prompt, schema, *, tag="", model=None):
            return [{"pattern_id": "p1", "root_cause": "r",
                     "failing_task_ids": ["t1"], "blamed_part": "tool_code"}]

    out = diagnoser.diagnose(_Backend(), [runner.Failure("t1", "task")])
    assert out[0]["verifier_cause"] == ""
    assert out[0]["agent_mechanism"] == ""
    assert out[0]["addressable"] is True
