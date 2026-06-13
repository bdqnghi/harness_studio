"""Structured failure evidence + a per-harness evidence store.

This is the data layer that makes context-localization possible. Each benchmark
adapter turns its raw verifier output (tau2's ``reward_info``, a coding bench's
test stdout, ...) into a benchmark-agnostic :class:`TaskEvidence`: the mechanical
checks that failed (:class:`VerifierSignal`) plus the *causal* transcript
windows those failures point at (:class:`TraceWindow`). Everything downstream —
the localizer (``stages/optimize/localizer.py``) and the editor (``strategist``) —
consumes only this structured form, so it never learns benchmark specifics.

The two design rules that make it useful where the old blind 4-message tail was
not:

* **Causal windows, not a tail.** ``select_windows`` indexes the transcript by
  *which check failed* (the tool call it names, the asserted string), so the
  evidence centers on the turn that actually went wrong — wherever in the
  dialogue it happened — and only falls back to the tail when nothing localizes.
* **Evidence survives to the editor.** The full (capped) transcript lives in the
  store in memory and is written out by ``materialize`` so a coding agent can be
  handed it as a read-only dir (``run_agent(read_dirs=...)``). It does not depend
  on a benchmark's scratch dir, which adapters routinely ``rmtree``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerifierSignal:
    """One mechanical check the verifier ran on a task's rollout."""

    kind: str  # "action" | "nl_assertion" | "db" | "communicate" | "test" | "other"
    name: str  # tool/assertion/test name (may be "" for unnamed checks)
    passed: bool
    detail: str = ""  # decoded mismatch / assertion text / db diff / error string


@dataclass
class TraceWindow:
    """A contiguous, cited slice of one task's transcript."""

    task_id: str
    trial: int
    start_idx: int
    end_idx: int  # inclusive
    messages: list[dict]  # role/content/tool_calls, per-message capped
    reason: str = ""  # why this window was selected


@dataclass
class TaskEvidence:
    """Everything localization needs about one failing task."""

    task_id: str
    reward: float
    trial: int = 0
    signals: list[VerifierSignal] = field(default_factory=list)
    windows: list[TraceWindow] = field(default_factory=list)
    transcript_len: int = 0  # total messages (truncation signaling)
    full_messages: list[dict] = field(default_factory=list)  # capped, for deep reads


# --- message helpers -------------------------------------------------------

def _content_str(m: dict) -> str:
    """Best-effort plain text of a message's content (str or list-of-parts)."""
    c = m.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for part in c:
            if isinstance(part, dict):
                out.append(str(part.get("text") or part.get("content") or part))
            else:
                out.append(str(part))
        return " ".join(out)
    return str(c)


def _tool_calls(m: dict) -> list[dict]:
    """Normalize tool calls from the shapes tau2 / OpenAI messages use."""
    calls = m.get("tool_calls") or m.get("tool_call") or []
    if isinstance(calls, dict):
        calls = [calls]
    return [c for c in calls if isinstance(c, dict)]


def _call_name(call: dict) -> str:
    fn = call.get("function") if isinstance(call.get("function"), dict) else None
    return str((fn or call).get("name", ""))


def _message_text(m: dict) -> str:
    """A searchable flattening of a message: role + content + tool-call names/args."""
    parts = [str(m.get("role", "")), _content_str(m)]
    for call in _tool_calls(m):
        parts.append(_call_name(call))
        args = (call.get("function") or call).get("arguments")
        if args is not None:
            parts.append(str(args))
    if m.get("name"):  # tool-result messages carry the tool name here
        parts.append(str(m["name"]))
    return " ".join(p for p in parts if p)


def _cap_msg(m: dict, cap: int) -> dict:
    """A trimmed copy safe to ship in a window (keeps role + capped content +
    tool-call names/args, which are the load-bearing evidence)."""
    out = {"role": m.get("role", "?"), "content": _content_str(m)[:cap]}
    names = [_call_name(c) for c in _tool_calls(m) if _call_name(c)]
    if names:
        out["tool_calls"] = names
    if m.get("name"):
        out["name"] = m["name"]
    return out


# --- causal window selection ----------------------------------------------

def _find_turn(messages: list[dict], signal: VerifierSignal) -> int | None:
    """Index of the LAST message that mentions the signal's name (the most
    likely site of a failed/expected action). ``None`` if it can't be
    localized — then the caller relies on the terminal window."""
    name = (signal.name or "").strip()
    if not name:
        return None
    found = None
    for i, m in enumerate(messages):
        if name in _message_text(m):
            found = i
    return found


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def select_windows(
    messages: list[dict],
    signals: list[VerifierSignal],
    *,
    task_id: str,
    trial: int = 0,
    radius: int = 1,
    per_msg_cap: int = 600,
    tail: int = 4,
) -> list[TraceWindow]:
    """Pick the transcript turns that the failed checks point at.

    For each failed signal, center a ±``radius`` window on the turn naming it;
    always include the terminal turn; merge overlaps. If no failed signal
    localizes to a turn, fall back to the last ``tail`` messages (never worse
    than the old behavior)."""
    n = len(messages)
    if n == 0:
        return []
    failed = [s for s in signals if not s.passed]
    centers: list[tuple[int, str]] = []
    for s in failed:
        idx = _find_turn(messages, s)
        if idx is not None:
            centers.append((idx, f"{s.kind} '{s.name}' failed: {s.detail}".strip()[:200]))

    if not centers:  # nothing localized → tail fallback
        start = max(0, n - tail)
        return [TraceWindow(
            task_id=task_id, trial=trial, start_idx=start, end_idx=n - 1,
            messages=[_cap_msg(m, per_msg_cap) for m in messages[start:]],
            reason="tail (no signal localized to a turn)",
        )]

    centers.append((n - 1, "final turn"))
    ranges = [(max(0, i - radius), min(n - 1, i + radius)) for i, _ in centers]
    windows: list[TraceWindow] = []
    for a, b in _merge_ranges(ranges):
        reasons = [r for i, r in centers if a <= i <= b and r != "final turn"]
        windows.append(TraceWindow(
            task_id=task_id, trial=trial, start_idx=a, end_idx=b,
            messages=[_cap_msg(m, per_msg_cap) for m in messages[a:b + 1]],
            reason="; ".join(reasons)[:300] or "final turn",
        ))
    return windows


def evidence_from_trace(
    task_id: str, reward: float, *, verifier_output: str = "",
    messages: list[dict] | None = None, failing_test: str = "", trial: int = 0,
    per_msg_cap: int = 600, max_full: int = 60,
) -> TaskEvidence:
    """Build :class:`TaskEvidence` for a coding-style benchmark (a test verifier
    + an agent trajectory) — the generalization seam used by the harbor adapters
    (nexau/mini-swe). The verifier output becomes a ``test`` signal; the
    trajectory messages give the causal window (localized to ``failing_test`` if
    it appears, else the tail)."""
    messages = messages or []
    signals: list[VerifierSignal] = []
    if reward < 1.0:
        signals.append(VerifierSignal(
            "test", failing_test or "", False, (verifier_output or "").strip()[-800:]))
    windows = select_windows(messages, signals, task_id=task_id, trial=trial,
                             per_msg_cap=per_msg_cap)
    full = [{"role": m.get("role", "?"), "content": _content_str(m)[:1000]}
            for m in messages[:max_full] if isinstance(m, dict)]
    return TaskEvidence(task_id=task_id, reward=reward, trial=trial, signals=signals,
                        windows=windows, transcript_len=len(messages), full_messages=full)


# --- rendering -------------------------------------------------------------

def _failed_summary(ev: TaskEvidence) -> str:
    failed = [s for s in ev.signals if not s.passed]
    if not failed:
        return ""
    return "; ".join(f"{s.kind}:{s.name} {s.detail}".strip() for s in failed)


def _window_text(w: TraceWindow) -> str:
    lines = [f"[messages {w.start_idx}-{w.end_idx}] {w.reason}"]
    for m in w.messages:
        tc = f" tool_calls={m['tool_calls']}" if m.get("tool_calls") else ""
        lines.append(f"[{m.get('role', '?')}]{tc} {m.get('content', '')}")
    return "\n".join(lines)


def to_flat_excerpt(ev: TaskEvidence, *, cap: int = 2400) -> str:
    """Render the legacy flat trace string (verifier reward + failed checks +
    the selected windows), so ``last_trace`` stays a drop-in for old callers."""
    parts = [f"reward={ev.reward}"]
    fs = _failed_summary(ev)
    if fs:
        parts.append("failed checks: " + fs[:500])
    for w in ev.windows:
        parts.append(_window_text(w))
    return "\n".join(parts)[:cap]


def evidence_brief(ev: TaskEvidence, *, cap: int) -> str:
    """Compact, editor-facing evidence for one task (failed checks + windows),
    budgeted to ``cap`` chars. This is what gets pushed inline to the editor;
    the full transcript is pull-only via the materialized dir."""
    return to_flat_excerpt(ev, cap=cap)


def render_evidence_md(ev: TaskEvidence) -> str:
    """Human/agent-readable evidence file written by ``EvidenceStore.materialize``."""
    out = [f"# Task {ev.task_id} (reward={ev.reward}, trial={ev.trial})", ""]
    failed = [s for s in ev.signals if not s.passed]
    if failed:
        out.append("## Failed verifier checks")
        for s in failed:
            out.append(f"- [{s.kind}] {s.name}: {s.detail}".rstrip())
        out.append("")
    if ev.windows:
        out.append("## Causal transcript windows")
        for w in ev.windows:
            out.append(_window_text(w))
            out.append("")
    if ev.full_messages:
        out.append(f"## Full transcript ({ev.transcript_len} messages)")
        for i, m in enumerate(ev.full_messages):
            tc = f" tool_calls={[_call_name(c) for c in _tool_calls(m)]}" if _tool_calls(m) else ""
            out.append(f"{i}. [{m.get('role', '?')}]{tc} {_content_str(m)[:600]}")
    return "\n".join(out) + "\n"


# --- store -----------------------------------------------------------------

def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:120]


class EvidenceStore:
    """Per-harness evidence, versioned by harness content hash (so a candidate's
    acceptance-run evidence is never attributed to the live harness). LRU over the
    most recent ``_MAX_HASHES`` harnesses — mirrors ``tau2._store_traces``."""

    _MAX_HASHES = 8

    def __init__(self) -> None:
        self._by_hash: dict[str, dict[str, TaskEvidence]] = {}

    def put(self, harness_hash: str, ev: TaskEvidence) -> None:
        bucket = self._by_hash.pop(harness_hash, {})
        self._by_hash[harness_hash] = bucket  # re-insert as most-recent
        bucket[ev.task_id] = ev
        while len(self._by_hash) > self._MAX_HASHES:
            self._by_hash.pop(next(iter(self._by_hash)))

    def get(self, harness_hash: str, task_id: str) -> TaskEvidence | None:
        return self._by_hash.get(harness_hash, {}).get(task_id)

    def all_for(self, harness_hash: str) -> dict[str, TaskEvidence]:
        return dict(self._by_hash.get(harness_hash, {}))

    def materialize(self, harness_hash: str, dest: Path) -> Path:
        """Write each task's evidence to ``dest/<task_id>.md`` and return ``dest``,
        suitable to hand to ``run_agent(read_dirs=[dest])``."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for tid, ev in self._by_hash.get(harness_hash, {}).items():
            (dest / f"{_safe(tid)}.md").write_text(render_evidence_md(ev))
        return dest
