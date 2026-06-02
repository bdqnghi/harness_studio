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
