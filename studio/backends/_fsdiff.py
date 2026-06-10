"""Shared: snapshot a workspace and diff to find changed files.

``files_changed`` is the one contract the Shell and Gate trust to enforce the
per-part edit budget and do-not-touch protection, so every Tier-A backend must
compute it byte-identically. Keeping the implementation here guarantees that.
"""

from __future__ import annotations

from pathlib import Path

_IGNORE = {"__pycache__", ".git", ".pytest_cache"}


def snapshot(root: Path) -> dict[str, str]:
    """Map ``relative_path -> file_text`` for every text file under ``root``."""
    root = Path(root)
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(part in _IGNORE for part in p.parts):
            try:
                out[str(p.relative_to(root))] = p.read_text(errors="replace")
            except OSError:
                pass
    return out


def diff(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Relative paths that were added, removed, or modified."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
