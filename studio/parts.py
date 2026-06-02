"""The seven editable harness part types and the part map.

A harness (PRD §2) is an open codebase whose editable components fall into seven
types. The Mapper (§5.0a) labels which files/regions implement each type; the
PartMap is that labeling. Everything not mapped is "do-not-touch".

In M0 the loop is *untyped* (it treats the whole workspace as one artifact) so
the PartMap is informational only. M1 onward uses it to enforce per-part edit
budgets and to tell the Strategist what it may change.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class PartType(str, enum.Enum):
    """The seven editable part types (PRD §1.4, §2)."""

    INSTRUCTIONS = "instructions"
    TOOL_DESCRIPTIONS = "tool_descriptions"
    TOOL_CODE = "tool_code"
    MIDDLEWARE = "middleware"
    SKILLS = "skills"
    SUBAGENTS = "subagents"
    MEMORY = "memory"

    @classmethod
    def all(cls) -> list["PartType"]:
        return list(cls)


# Sentinel a Mapper uses when a part type is not present in a codebase.
ABSENT = "absent"


def _coerce_paths(value) -> list[str]:
    """Normalize a Mapper field to a list of path strings, tolerating malformed
    AI output (``"absent"``, ``None``, a bare string, or junk)."""
    if value is None or value == ABSENT:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []  # unexpected type (int/dict/...) -> treat as absent


@dataclass
class PartMap:
    """Labels each part type with the files (relative paths) that implement it.

    ``parts[PartType.X]`` is a list of relative paths, or empty if absent.
    ``do_not_touch`` is every other file the optimizer must not edit.
    """

    parts: dict[PartType, list[str]] = field(default_factory=dict)
    do_not_touch: list[str] = field(default_factory=list)

    def files_for(self, part: PartType) -> list[str]:
        return list(self.parts.get(part, []))

    def is_present(self, part: PartType) -> bool:
        return bool(self.parts.get(part))

    def editable_files(self) -> list[str]:
        """All files the optimizer is allowed to change, de-duplicated."""
        seen: list[str] = []
        for paths in self.parts.values():
            for p in paths:
                if p not in seen:
                    seen.append(p)
        return seen

    def part_of(self, path: str) -> PartType | None:
        """Which part type a given file belongs to, or None if do-not-touch."""
        for part, paths in self.parts.items():
            if path in paths:
                return part
        return None

    # --- serialization (round-trips through the Mapper's JSON output) ---

    def to_dict(self) -> dict:
        out: dict = {p.value: (self.parts.get(p) or ABSENT) for p in PartType}
        out["do_not_touch"] = list(self.do_not_touch)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "PartMap":
        parts: dict[PartType, list[str]] = {p: _coerce_paths(data.get(p.value)) for p in PartType}
        return cls(parts=parts, do_not_touch=_coerce_paths(data.get("do_not_touch")))
