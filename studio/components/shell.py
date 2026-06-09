"""Code shell (PRD §5.6): enforce the hard invariants the AI cannot be trusted with.

Two reliable, code-checkable rules:
  * **Do-not-touch protection** — any change to a file outside the editable part
    map (including a newly-created file) is reverted to the original. The
    Strategist only ever gets to change real, mapped parts.
  * **Per-part edit budget** — at most ``budget_per_part`` changed files per part
    type; exceeding it rejects the whole strategy (recorded to the avoid-list).

Broken references (one edit removing a name another file needs) are *not* checked
here — they surface for free at the structural check (§5.7), which boots the
candidate. Keeping the shell to invariants it can verify reliably avoids fragile
heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..harness import Harness
from ..parts import PartMap, PartType


@dataclass
class ShellResult:
    ok: bool
    changed_parts: dict[PartType, list[str]] = field(default_factory=dict)
    reverted: list[str] = field(default_factory=list)  # do-not-touch files restored
    violations: list[str] = field(default_factory=list)


def _changed_files(original: Harness, candidate: Harness) -> list[str]:
    orig = set(original.files())
    cand = set(candidate.files())
    changed = sorted(orig | cand)
    out = []
    for rel in changed:
        if rel not in orig or rel not in cand:
            out.append(rel)  # added or removed
        elif original.read_file(rel) != candidate.read_file(rel):
            out.append(rel)  # modified
    return out


def enforce(
    original: Harness,
    candidate: Harness,
    part_map: PartMap,
    *,
    budget_per_part: int = 3,
) -> ShellResult:
    """Validate (and minimally repair) a candidate's edits in place."""
    reverted: list[str] = []

    # 1. Revert any change to a do-not-touch / unmapped file. A part entry ending
    #    in "/" is a directory: new files created under it (e.g. a new tool) are
    #    editable, so the optimizer can ADD capabilities, not just edit existing
    #    files (matching AHE's freedom).
    for rel in _changed_files(original, candidate):
        if not part_map.is_editable(rel):
            _revert(original, candidate, rel)
            reverted.append(rel)

    # 2. Bucket the surviving changes by part and enforce the budget.
    changed_parts: dict[PartType, list[str]] = {}
    for rel in _changed_files(original, candidate):
        part = part_map.part_of(rel)
        if part is not None:
            changed_parts.setdefault(part, []).append(rel)

    violations = [
        f"part {part.value} changed {len(files)} files (budget {budget_per_part})"
        for part, files in changed_parts.items()
        if len(files) > budget_per_part
    ]
    return ShellResult(
        ok=not violations,
        changed_parts=changed_parts,
        reverted=reverted,
        violations=violations,
    )


def _revert(original: Harness, candidate: Harness, rel: str) -> None:
    """Restore ``rel`` in the candidate to its original state (or delete it if
    the original had no such file)."""
    if original.exists(rel):
        candidate.write_file(rel, original.read_file(rel))
    else:
        (candidate.root / rel).unlink(missing_ok=True)
