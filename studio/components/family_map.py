"""The strategy-family map (PRD §5.10.2) — the interface between the two loops.

A durable markdown file with four sections. The inner loop *reads* it every round
(the Strategist prefers 'works' families and avoids 'falsified' ones); the outer
loop *rewrites* it every segment (the Meta-agent, or cheap rules). The grain is
**families** — classes of approach (e.g. "instructions+tool_code") — because a
lesson at that grain generalizes, while "don't repeat edit #47" does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_SECTIONS = [
    ("works", "Works (prefer)"),
    ("falsified", "Falsified (do not repeat)"),
    ("pivot", "Pivot toward"),
    ("open", "Open / untried"),
]
_TITLE_TO_KEY = {title: key for key, title in _SECTIONS}


@dataclass
class FamilyMap:
    works: list[str] = field(default_factory=list)
    falsified: list[str] = field(default_factory=list)
    pivot: list[str] = field(default_factory=list)
    open: list[str] = field(default_factory=list)

    # --- text round-trip ---

    def to_text(self) -> str:
        out = ["# Strategy-family map", ""]
        for key, title in _SECTIONS:
            out.append(f"## {title}")
            items = getattr(self, key)
            out.extend(f"- {it}" for it in items) if items else out.append("- (none)")
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    @classmethod
    def from_text(cls, text: str) -> "FamilyMap":
        fm = cls()
        current: list[str] | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("## "):
                key = _TITLE_TO_KEY.get(line[3:].strip())
                current = getattr(fm, key) if key else None
            elif line.startswith("- ") and current is not None:
                item = line[2:].strip()
                if item and item != "(none)":
                    current.append(item)
        return fm

    # --- persistence ---

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_text())

    @classmethod
    def load(cls, path: Path) -> "FamilyMap":
        path = Path(path)
        return cls.from_text(path.read_text()) if path.exists() else cls()

    # --- mutation (used by the rule-based updater and the meta-agent) ---

    def _family_names(self, items: list[str]) -> set[str]:
        return {it.split(":", 1)[0].strip() for it in items}

    def promote(self, family: str, why: str) -> None:
        if family in self._family_names(self.works):
            return
        self.works.append(f"{family}: {why}")
        self.open = [it for it in self.open if it.split(":", 1)[0].strip() != family]

    def falsify(self, family: str, reason: str) -> None:
        if family in self._family_names(self.falsified):
            return
        self.falsified.append(f"{family}: {reason}")
        # A falsified family leaves the works/open frontier.
        self.works = [it for it in self.works if it.split(":", 1)[0].strip() != family]
        self.open = [it for it in self.open if it.split(":", 1)[0].strip() != family]

    def add_pivot(self, directive: str) -> None:
        if directive not in self.pivot:
            self.pivot.append(directive)

    def do_not_repeat(self) -> list[str]:
        """Family names the Strategist must avoid (for the Reviewer)."""
        return sorted(self._family_names(self.falsified))


def init_map(path: Path) -> FamilyMap:
    """Create an empty four-section map on disk (PRD §5.0d)."""
    fm = FamilyMap()
    fm.save(path)
    return fm
