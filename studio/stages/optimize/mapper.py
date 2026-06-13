"""Mapper (PRD §5.0a): label the harness files into the seven editable parts.

This defines the optimization search space, so it bounds everything downstream
(PRD §1.5, §12). It is a Tier-B call: a bounded codebase listing in, a part map
out. Re-run at segment boundaries because the codebase changes as edits land;
any unmapped file is do-not-touch.
"""

from __future__ import annotations

from studio import schemas
from studio.backends.base import Backend
from studio.core.harness import Harness
from studio.core.parts import ABSENT, PartMap, PartType

TAG = "mapper"

_PART_GUIDE = {
    PartType.INSTRUCTIONS: "system prompt / agent instructions (prose that steers the model)",
    PartType.TOOL_DESCRIPTIONS: "tool schemas / descriptions the model reads when choosing tools",
    PartType.TOOL_CODE: "tool implementations (the functions tools actually run)",
    PartType.MIDDLEWARE: "request/response hooks: retries, caching, truncation, summarization",
    PartType.SKILLS: "on-demand skill/knowledge packages (e.g. SKILL.md files)",
    PartType.SUBAGENTS: "sub-agent / delegated-agent configuration",
    PartType.MEMORY: "persistent long-term memory files",
}


def build_listing(harness: Harness, head_lines: int = 40) -> str:
    """A compact codebase view: the file tree plus the head of each text file."""
    lines = ["# File tree", *(f"- {f}" for f in harness.files()), "", "# File heads"]
    for rel in harness.files():
        try:
            text = harness.read_file(rel)
        except (UnicodeDecodeError, OSError):
            lines.append(f"\n## {rel}\n(binary or unreadable)")
            continue
        head = "\n".join(text.splitlines()[:head_lines])
        lines.append(f"\n## {rel}\n{head}")
    return "\n".join(lines)


def _prompt(listing: str) -> str:
    guide = "\n".join(f"- {p.value}: {desc}" for p, desc in _PART_GUIDE.items())
    return (
        "You are labeling an AI agent's codebase (its 'harness') into seven editable "
        "part types. For each part type, list the RELATIVE file paths that implement it "
        f'in THIS codebase, or the string "{ABSENT}" if the codebase has no such part.\n\n'
        f"Part types:\n{guide}\n\n"
        "A single file may implement more than one part (e.g. tool descriptions and "
        "tool code can live in one module — list it under both). Put every file that is "
        "NOT an editable part (entry points, tests, packaging, vendored code) into "
        '"do_not_touch". Use paths exactly as shown in the tree.\n\n'
        f"{listing}"
    )


def map_harness(
    backend: Backend, harness: Harness, *, model: str | None = None, head_lines: int = 40
) -> PartMap:
    data = backend.prompt_json(
        _prompt(build_listing(harness, head_lines)), schemas.PART_MAP, tag=TAG, model=model
    )
    part_map = PartMap.from_dict(data)
    return _restrict_to_existing(part_map, harness)


def _restrict_to_existing(part_map: PartMap, harness: Harness) -> PartMap:
    """Drop labels for files that don't exist (model hallucinations) and resolve
    overlaps, so the rest of the pipeline only sees real, unambiguous paths."""
    real = set(harness.files())
    for part in PartType:
        part_map.parts[part] = [p for p in part_map.parts.get(part, []) if p in real]
    editable = set(part_map.editable_files())
    # A file mapped to a part is editable; it must not also be do-not-touch.
    part_map.do_not_touch = [
        p for p in part_map.do_not_touch if p in real and p not in editable
    ]
    return part_map
