"""The Harness: an open codebase laid out as files in a directory.

This is the object being optimized (PRD §2). It is deliberately thin — a harness
*is* a directory of text files — so the same type works for the toy target and
for a real one (Terminus-KIRA). All mutation happens by editing files on disk;
the Strategist (a coding agent) or the MockBackend writes those edits, and the
acceptance is the only thing that promotes a candidate to be the live harness.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

# Directories never copied/hashed as part of the harness content.
_IGNORE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


@dataclass
class Harness:
    """A harness rooted at a directory of editable files."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    # --- file access ---

    def files(self) -> list[str]:
        """Relative paths of all content files, sorted for determinism."""
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not any(part in _IGNORE_DIRS for part in p.parts):
                out.append(str(p.relative_to(self.root)))
        return out

    def read_file(self, rel: str) -> str:
        return (self.root / rel).read_text()

    def write_file(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def exists(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    # --- copying / identity ---

    def copy_to(self, dest: Path) -> "Harness":
        """Copy the whole tree to ``dest`` and return a Harness over it."""
        dest = Path(dest)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(
            self.root, dest, ignore=shutil.ignore_patterns(*_IGNORE_DIRS)
        )
        return Harness(dest)

    def content_hash(self) -> str:
        """Stable hash over all file contents — used for caching and as the
        deterministic seed for the toy benchmark's injected noise_floor."""
        h = hashlib.sha256()
        for rel in self.files():
            h.update(rel.encode())
            h.update(b"\0")
            h.update((self.root / rel).read_bytes())
            h.update(b"\0")
        return h.hexdigest()
