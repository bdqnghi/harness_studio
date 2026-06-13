"""Localizer: evidence-grounded, span-level edit targets (the missing step).

Given this round's failure *patterns* (from the diagnoser), the failing-task
*evidence* (materialized by the :class:`EvidenceStore`), and the *editable*
harness files, the localizer produces concrete :data:`schemas.LOCALIZATION`
targets: which editable file + which span/rule to change, each CITING the exact
transcript evidence and the exact current harness text it would edit.

This is SHO's analog of how Claude Code localizes: an iterative read-only pass
that returns a compact, evidence-cited conclusion — plus Claude Code's hard rule
"don't change code you haven't read". Here that rule is enforced deterministically
by a **citation guard**: a target is dropped unless its ``current_text`` and at
least one evidence ``quote`` are verbatim (whitespace-normalized) substrings of
what was actually read. A hallucinated localization can never reach the editor.

Two modes (the "delegation scales to difficulty" lesson):

* **inline** — one ``prompt_json`` call with the evidence + editable files inlined.
  Right for a small, single-file editable surface (e.g. tau2's ``policy.md``).
* **agentic** — a read-only ``run_explore`` loop (grep/read over the harness +
  evidence dirs). Right for multi-file coding harnesses or many failure clusters.

``localize`` never raises: any failure (including a backend without
``run_explore``) degrades — agentic falls back to inline, and a total failure
returns ``[]`` so the round proceeds on diagnosis alone.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import schemas
from ..backends.base import Backend
from ..harness import Harness

TAG = "localizer"

_CORPUS_AGENTIC_THRESHOLD = 12_000  # evidence chars above which we go agentic
_INLINE_CORPUS_CAP = 8_000
_INLINE_FILES_CAP = 8_000


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _corpus_files(evidence_dir: Path) -> list[Path]:
    d = Path(evidence_dir)
    return sorted(d.glob("*.md")) if d.is_dir() else []


def _read_corpus(evidence_dir: Path, *, cap: int) -> str:
    parts = []
    for f in _corpus_files(evidence_dir):
        try:
            parts.append(f.read_text(errors="replace"))
        except OSError:
            continue
    return "\n\n".join(parts)[:cap]


def _editable_contents(harness: Harness, editable_files: list[str], *, cap: int) -> str:
    out = []
    for f in editable_files:
        if f.endswith("/"):
            continue  # directory entry, not a file
        try:
            body = harness.read_file(f) if harness.exists(f) else "(missing)"
        except Exception:  # noqa: BLE001
            body = "(unreadable)"
        out.append(f"### {f}\n{body}")
    return "\n\n".join(out)[:cap]


def _patterns_text(patterns: list[dict]) -> str:
    lines = []
    for p in patterns:
        lines.append(
            f"- [{p.get('pattern_id', '?')}] {p.get('root_cause', p.get('description', ''))} "
            f"(verifier: {p.get('verifier_cause', '')}; agent: {p.get('agent_mechanism', '')}; "
            f"tasks: {', '.join(p.get('failing_task_ids', [])[:5])})"
        )
    return "\n".join(lines)


def choose_mode(mode: str, patterns: list[dict], editable_files: list[str],
                evidence_dir: Path) -> str:
    """Pick inline vs agentic. Explicit modes pass through; 'auto' goes agentic
    when the surface is multi-file, the failures are many, or the evidence is
    large — otherwise inline."""
    if mode in ("inline", "agentic"):
        return mode
    real_files = [f for f in editable_files if not f.endswith("/")]
    if len([f for f in editable_files if f.endswith("/")]) or len(real_files) > 1:
        return "agentic"
    if len(patterns) >= 3:
        return "agentic"
    if len(_read_corpus(evidence_dir, cap=_CORPUS_AGENTIC_THRESHOLD + 1)) > _CORPUS_AGENTIC_THRESHOLD:
        return "agentic"
    return "inline"


def _common_tail() -> str:
    return (
        "Return targets: for each give pattern_id, target_file (one of the editable "
        "files ONLY), target_locator (the line range or the rule/section heading), "
        "current_text (the EXACT current text you would change, copied verbatim — "
        "leave empty ONLY for a pure addition), change_kind, rationale, confidence, "
        "and evidence[] each with task_id + a quote copied VERBATIM from the failure "
        "evidence (optionally signal, msg_range). Do not paraphrase quotes or "
        "current_text — they are checked against the source and dropped if invented."
    )


def _inline(backend, patterns, harness, evidence_dir, editable_files, model) -> dict:
    prompt = (
        "Localize the cause of these failures to a specific span/rule of a specific "
        "editable harness file, citing exact evidence.\n\n"
        f"Editable files (target ONLY these): {', '.join(editable_files)}\n\n"
        f"Current editable file contents:\n{_editable_contents(harness, editable_files, cap=_INLINE_FILES_CAP)}\n\n"
        f"Failure patterns:\n{_patterns_text(patterns)}\n\n"
        f"Failure evidence (verifier checks + causal transcript windows):\n"
        f"{_read_corpus(evidence_dir, cap=_INLINE_CORPUS_CAP)}\n\n"
        + _common_tail()
    )
    return backend.prompt_json(prompt, schemas.LOCALIZATION, tag=TAG, model=model)


def _agentic(backend, patterns, harness, evidence_dir, editable_files, model) -> dict:
    instruction = (
        "Localize the cause of the failing tasks. The evidence directory holds "
        "per-task failure files (verifier checks + causal transcript windows); the "
        "harness directory holds the editable files. Use read_file/list_dir/grep to "
        "find the EXACT span to change.\n\n"
        f"Editable files (target ONLY these): {', '.join(editable_files)}\n\n"
        f"Failure patterns:\n{_patterns_text(patterns)}\n\n"
        + _common_tail()
    )
    return backend.run_explore(
        instruction, read_dirs=[Path(harness.root), Path(evidence_dir)],
        schema=schemas.LOCALIZATION, tag=TAG, model=model,
    )


def _validate(targets, harness, evidence_dir, editable_files) -> list[dict]:
    """Citation guard: keep only targets that name an editable file, whose
    ``current_text`` (if any) is a verbatim substring of that file, and that
    cite at least one evidence quote present in the materialized corpus.
    Non-matching quotes are stripped."""
    editable = set(editable_files)
    corpus = _norm(_read_corpus(evidence_dir, cap=10_000_000))
    file_norm: dict[str, str] = {}
    for f in editable_files:
        try:
            file_norm[f] = _norm(harness.read_file(f)) if harness.exists(f) else ""
        except Exception:  # noqa: BLE001
            file_norm[f] = ""

    kept = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        tf = t.get("target_file", "")
        if tf not in editable:
            continue
        ct = t.get("current_text", "")
        if ct and _norm(ct) not in file_norm.get(tf, ""):
            continue  # claims an edit site that isn't actually in the file
        grounded = [
            e for e in (t.get("evidence") or [])
            if isinstance(e, dict) and e.get("quote") and _norm(e["quote"]) in corpus
        ]
        if not grounded:
            continue  # no grounded evidence -> not trustworthy
        t = dict(t)
        t["evidence"] = grounded
        kept.append(t)
    return kept


def localize(backend: Backend, patterns: list[dict], harness: Harness,
             evidence_dir: Path, *, editable_files: list[str],
             mode: str = "auto", model: str | None = None) -> list[dict]:
    """Return validated, evidence-cited localization targets (possibly empty)."""
    if not patterns or not editable_files:
        return []
    chosen = choose_mode(mode, patterns, editable_files, evidence_dir)
    raw = None
    if chosen == "agentic":
        try:
            raw = _agentic(backend, patterns, harness, evidence_dir, editable_files, model)
        except Exception:  # noqa: BLE001 — incl. NotImplementedError -> fall back to inline
            raw = None
    if raw is None:
        try:
            raw = _inline(backend, patterns, harness, evidence_dir, editable_files, model)
        except Exception:  # noqa: BLE001 — localization is a hint, never a hard dependency
            return []
    targets = raw.get("targets", []) if isinstance(raw, dict) else []
    return _validate(targets, harness, evidence_dir, editable_files)
