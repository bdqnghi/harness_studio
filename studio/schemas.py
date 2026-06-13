"""JSON schemas for AI-helper outputs, plus a tiny stdlib-only validator.

Every Tier-B helper (Mapper, Diagnoser, Reviewer, Ranker) returns JSON that the
Orchestrator must validate before trusting it (PRD §5.12: "validate every AI
output ... retry-once-or-skip on malformed output"). We avoid a third-party
``jsonschema`` dependency by implementing the small subset of JSON Schema we
actually use. The same schema objects are passed to ``claude -p --json-schema``
so the model is steered toward valid output in the first place.
"""

from __future__ import annotations

from typing import Any


class SchemaError(ValueError):
    """Raised when data does not conform to a schema."""


def validate(data: Any, schema: dict, _path: str = "$") -> None:
    """Validate ``data`` against ``schema``; raise SchemaError on mismatch.

    Supports: type (object/array/string/integer/number/boolean), properties,
    required, items, enum, additionalProperties=False. That is all our helper
    schemas need — kept deliberately small and readable.
    """
    if "anyOf" in schema:
        for sub in schema["anyOf"]:
            try:
                validate(data, sub, _path)
                break
            except SchemaError:
                continue
        else:
            raise SchemaError(f"{_path}: {data!r} matched none of anyOf")
        return

    t = schema.get("type")
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(f"{_path}: {data!r} not in enum {schema['enum']}")

    if t == "object":
        if not isinstance(data, dict):
            raise SchemaError(f"{_path}: expected object, got {type(data).__name__}")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in data:
                raise SchemaError(f"{_path}: missing required key '{key}'")
        if schema.get("additionalProperties") is False:
            extra = set(data) - set(props)
            if extra:
                raise SchemaError(f"{_path}: unexpected keys {sorted(extra)}")
        for key, value in data.items():
            if key in props:
                validate(value, props[key], f"{_path}.{key}")

    elif t == "array":
        if not isinstance(data, list):
            raise SchemaError(f"{_path}: expected array, got {type(data).__name__}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(data):
                validate(item, item_schema, f"{_path}[{i}]")

    elif t == "string":
        if not isinstance(data, str):
            raise SchemaError(f"{_path}: expected string, got {type(data).__name__}")
    elif t == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            raise SchemaError(f"{_path}: expected integer")
    elif t == "number":
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise SchemaError(f"{_path}: expected number")
    elif t == "boolean":
        if not isinstance(data, bool):
            raise SchemaError(f"{_path}: expected boolean")


# --- Concrete schemas (filled in as each helper is added per milestone) ---

# Diagnoser (§5.2): cluster failures into causes, blame a part. The signature
# triple (verifier_cause, agent_mechanism, addressable) follows Self-Harness
# (2606.09498): structure lives on the *failure*, not the edit. The fields are
# optional in the schema (a model omitting them must not fail the round);
# ``diagnoser.diagnose`` default-fills them so downstream code can rely on them.
DIAGNOSIS = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["pattern_id", "root_cause", "failing_task_ids", "blamed_part"],
        "properties": {
            "pattern_id": {"type": "string"},
            "description": {"type": "string"},
            "root_cause": {"type": "string"},
            "failing_task_ids": {"type": "array", "items": {"type": "string"}},
            "blamed_part": {"type": "string"},
            "confidence": {"type": "number"},
            "verifier_cause": {"type": "string"},    # what the verifier observed
            "agent_mechanism": {"type": "string"},   # what the agent did to cause it
            "addressable": {"type": "boolean"},      # fixable by editing the harness?
        },
    },
}

# Mapper (§5.0a): label files into the seven part types.
PART_MAP = {
    "type": "object",
    "required": [
        "instructions", "tool_descriptions", "tool_code", "middleware",
        "skills", "subagents", "memory", "do_not_touch",
    ],
    "properties": {
        # each part is either a list of paths or the literal string "absent"
        **{
            name: {"anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"enum": ["absent"]},  # parts.ABSENT
            ]}
            for name in [
                "instructions", "tool_descriptions", "tool_code", "middleware",
                "skills", "subagents", "memory",
            ]
        },
        "do_not_touch": {"type": "array", "items": {"type": "string"}},
    },
}

# Reviewer (§5.4): keep / drop whole strategies.
REVIEW = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keep", "drop"],
    "properties": {
        "keep": {"type": "array", "items": {"type": "string"}},
        "drop": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["strategy_id", "reason"],
                "properties": {
                    "strategy_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        },
    },
}

# Ranker (§5.5): ordered list of strategy ids, best first.
RANKING = {
    "type": "object",
    "additionalProperties": False,
    "required": ["order"],
    "properties": {"order": {"type": "array", "items": {"type": "string"}}},
}

# Direction router (tree optimizer): assign each failure pattern to an existing
# direction node or propose a new one ("" direction_id => create new).
DIRECTION_ASSIGN = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assignments"],
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pattern_id", "direction_id"],
                "properties": {
                    "pattern_id": {"type": "string"},
                    "direction_id": {"type": "string"},
                    "new_title": {"type": "string"},
                    "new_mechanism": {"type": "string"},
                },
            },
        },
    },
}

# Ideator (tree optimizer): k cheap text hypotheses under a direction — the
# Arbor Mechanism/Hypothesis/Observable/Conflicts discipline.
HYPOTHESES = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "mechanism", "hypothesis", "observable"],
                "properties": {
                    "title": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "observable": {"type": "string"},
                    "conflicts": {"type": "string"},
                },
            },
        },
    },
}

# Insight distiller (tree optimizer): the <=200-word lesson from one test.
INSIGHT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["insight"],
    "properties": {"insight": {"type": "string"}},
}

# Localization: evidence-grounded, span/rule-level edit targets (stages/optimize/
# localizer.py). Each target names the editable file + the span to change and
# cites the exact evidence (transcript quote) and current harness text. The
# orchestrator validates current_text / quotes are real substrings before trust.
LOCALIZATION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["targets"],
    "properties": {
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pattern_id", "target_file", "evidence"],
                "properties": {
                    "pattern_id": {"type": "string"},
                    "target_file": {"type": "string"},
                    "target_locator": {"type": "string"},   # "lines 40-52" or a rule/section
                    "current_text": {"type": "string"},      # exact span to change (read-before-act)
                    "change_kind": {"type": "string"},        # modify_rule|add_rule|fix_code|add_tool|...
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["task_id", "quote"],
                            "properties": {
                                "task_id": {"type": "string"},
                                "signal": {"type": "string"},
                                "quote": {"type": "string"},
                                "msg_range": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
}

# (Cold-start no longer uses a fill schema: the coding agent generates the harness
#  from scratch via strategist.build_harness — see stages/optimize/strategist.py.)
