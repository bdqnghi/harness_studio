"""Known-good (and known-bad) edit actions for the toy harness.

These encode the toy's *known optimum*: applying the three good fixes takes the
harness from the baseline (only ``echo`` works) to a perfect score. They are
``MockBackend`` Tier-A actions — each mutates files under a workspace root — so
tests and ``run_toy.py`` can script an exact proposer sequence and assert the
acceptance keeps the good edits and rejects the bad ones.
"""

from __future__ import annotations

from pathlib import Path


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    path = Path(root) / rel
    text = path.read_text()
    if old not in text:
        raise AssertionError(f"toy fix: pattern not found in {rel}: {old!r}")
    path.write_text(text.replace(old, new, 1))


# --- good fixes (one per failure family) ---

def fix_reverse(root: Path) -> None:
    """Blame: tool_code. Make _reverse actually reverse."""
    _edit(root, "tools.py", "    return arg  # BUG: should reverse the string",
          "    return arg[::-1]")


def enable_upper(root: Path) -> None:
    """Blame: instructions. Turn on the already-correct upper op."""
    path = Path(root) / "instructions.txt"
    path.write_text(path.read_text().rstrip() + "\nENABLE upper\n")


def implement_add(root: Path) -> None:
    """Blame: tool_code. Add the missing add implementation."""
    _edit(root, "tools.py", '    # "add" is intentionally missing.',
          '    "add": _add,')
    _edit(root, "tools.py", "OPS = {",
          "def _add(arg):\n    a, b = arg.split(\",\")\n    return str(int(a) + int(b))\n\n\nOPS = {")


def enable_add(root: Path) -> None:
    """Blame: instructions. Turn on the add op (pairs with implement_add)."""
    path = Path(root) / "instructions.txt"
    path.write_text(path.read_text().rstrip() + "\nENABLE add\n")


def fix_add_full(root: Path) -> None:
    """Coordinated multi-part fix: implement + enable add together."""
    implement_add(root)
    enable_add(root)


# --- bad edits (the acceptance / structural check must reject these) ---

def noop(root: Path) -> None:
    """A harmless edit that changes nothing meaningful (acceptance: no help)."""
    path = Path(root) / "config.json"
    path.write_text(path.read_text())  # rewrite identical content


def enable_bogus(root: Path) -> None:
    """A real edit that changes a file but does not affect any score: enable an
    operation that no task uses (acceptance: no help -> reject, but reaches the acceptance)."""
    path = Path(root) / "instructions.txt"
    path.write_text(path.read_text().rstrip() + "\nENABLE bogus\n")


def regress_echo(root: Path) -> None:
    """Disable the working echo op (acceptance: regression -> reject)."""
    path = Path(root) / "instructions.txt"
    path.write_text(path.read_text().replace("ENABLE echo\n", ""))


def break_boot(root: Path) -> None:
    """Write invalid Python (structural check must catch before the acceptance)."""
    (Path(root) / "tools.py").write_text("OPS = {  # truncated, syntax error\n")
