"""Diagnoser (PRD §5.2): turn raw failures into causes + blame.

Two paths:

* **structured** (preferred) — when the runner produced ``FailureSignal``s
  (benchmark exposes ``last_evidence``): pre-group failures *deterministically*
  by their failed-check signature, then a Tier-B call only NAMES each group
  (root cause, blamed part, addressable). Membership + ``tasks_affected`` counts
  are deterministic and survive an LLM failure — the engine always returns
  patterns. Each pattern's ``verifier_cause`` is the literal failed-check set
  (ground truth), not an LLM guess.
* **legacy** — when only the flat ``last_trace`` is available (toy/kira): the
  original single-call clustering over text excerpts.

Routing rides on the same output (no separate router). Blame is a hint; the
acceptance still decides, so a wrong blame just wastes one strategy.
"""

from __future__ import annotations

from studio import schemas
from studio.backends.base import Backend
from studio.core.parts import PartType
from studio.stages.optimize.diagnose.runner import Failure
from studio.stages.optimize.diagnose.signals import FailureSignal, signature

TAG = "diagnoser"

_BLAME_OPTIONS = [p.value for p in PartType] + ["unclear"]


def diagnose(
    backend: Backend,
    failures: list[Failure],
    *,
    records: list[FailureSignal] | None = None,
    model: str | None = None,
) -> list[dict]:
    """Cluster failures into patterns. Uses the structured path when ``records``
    are present, else the legacy flat-trace path."""
    if records:
        return _diagnose_structured(backend, records, model=model)
    return _diagnose_legacy(backend, failures, model=model)


# --- structured (verifier-grounded) ---------------------------------------

def _check_label(kind: str, name: str) -> str:
    return f"{kind}:{name}" if name else kind


def _diagnose_structured(
    backend: Backend, records: list[FailureSignal], *, model: str | None = None
) -> list[dict]:
    # Deterministic pre-group by failed-check signature. Consistent failures
    # first; flaky/mixed ones are grouped separately so they don't dilute a mode.
    groups: dict[frozenset, list[FailureSignal]] = {}
    for r in records:
        groups.setdefault(signature(r), []).append(r)
    group_list = list(groups.items())  # stable insertion order

    blocks = []
    for i, (sig, recs) in enumerate(group_list):
        checks = ", ".join(_check_label(k, n) for k, n in sorted(sig)) or "unknown"
        diffs = sorted({d for r in recs for d in r.gt_diff})[:5]
        flaky = sum(1 for r in recs if not r.consistent)
        tasks = ", ".join(r.task_id for r in recs[:8])
        block = (f"g{i}: failed checks [{checks}] — {len(recs)} tasks"
                 f"{f' ({flaky} flaky)' if flaky else ''}: {tasks}")
        if diffs:
            block += "\n  ground-truth diffs:\n" + "\n".join(f"    - {d}" for d in diffs)
        if recs[0].window:
            block += "\n  sample trace:\n    " + recs[0].window[:600].replace("\n", "\n    ")
        blocks.append(block)

    prompt = (
        "Failing benchmark tasks have been pre-grouped by which verifier checks "
        "they failed (the ground truth). For EACH group g#, name the failure mode, "
        "infer its root cause, and blame the harness part most likely responsible. "
        "Do not change group membership.\n\n"
        f"Blame must be one of: {', '.join(_BLAME_OPTIONS)}.\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn a JSON array with one item per group: group_id, name, "
        "description, root_cause, blamed_part, agent_mechanism (what the agent did "
        "or failed to do), addressable (boolean; false when no harness edit could "
        "plausibly fix it — task-specific difficulty / flake / model-capability "
        "limit), confidence (0-1)."
    )
    try:
        names = backend.prompt_json(prompt, schemas.PATTERN_NAMES, tag=TAG, model=model)
    except Exception:  # noqa: BLE001 — never let naming kill the round; counts are deterministic
        names = []
    by_gid = {str(n.get("group_id")): n for n in names if isinstance(n, dict)}

    out = []
    for i, (sig, recs) in enumerate(group_list):
        gid = f"g{i}"
        n = by_gid.get(gid, {})
        checks = "; ".join(_check_label(k, nm) for k, nm in sorted(sig)) or "unknown"
        out.append({
            "pattern_id": gid,
            "description": n.get("name") or n.get("description", ""),
            "root_cause": n.get("root_cause") or checks,
            "failing_task_ids": [r.task_id for r in recs],          # DETERMINISTIC
            "tasks_affected": len(recs),                            # DETERMINISTIC count
            "blamed_part": n.get("blamed_part", "unclear"),
            "confidence": float(n.get("confidence", 0.5) or 0.5),
            "verifier_cause": checks,                               # ground truth, not guessed
            "agent_mechanism": n.get("agent_mechanism", ""),
            "addressable": bool(n.get("addressable", True)),
        })
    return out


# --- legacy (flat-trace) ---------------------------------------------------

def _diagnose_legacy(
    backend: Backend, failures: list[Failure], *, model: str | None = None
) -> list[dict]:
    if not failures:
        return []

    def _entry(f: Failure) -> str:
        head = f"- {f.task_id}: {f.description}"
        if f.trace:
            body = "\n".join("    " + ln for ln in f.trace.splitlines())
            return f"{head}\n  failure evidence:\n{body}"
        return head

    listing = "\n".join(_entry(f) for f in failures)
    prompt = (
        "These benchmark tasks failed. Use the failure evidence (verifier output "
        "and the agent's last actions) to cluster them by failure mode, infer the "
        "root cause of each cluster, and blame the harness part most likely "
        "responsible.\n\n"
        f"Blame must be one of: {', '.join(_BLAME_OPTIONS)}.\n\n"
        f"Failing tasks:\n{listing}\n\n"
        "Return a JSON array of clusters; each cluster has: pattern_id, "
        "description, root_cause, failing_task_ids (subset of the ids above), "
        "blamed_part, confidence (0-1), and a failure signature: verifier_cause "
        "(what the verifier mechanically observed), agent_mechanism (what the "
        "agent did or failed to do that produced it), addressable (boolean; "
        "false when no harness edit could plausibly fix it — task-specific "
        "difficulty, infrastructure flake, or raw model-capability limits)."
    )
    out = backend.prompt_json(prompt, schemas.DIAGNOSIS, tag=TAG, model=model)
    for d in out:
        if isinstance(d, dict):  # default-fill so downstream code can rely on the keys
            d.setdefault("verifier_cause", "")
            d.setdefault("agent_mechanism", "")
            d.setdefault("addressable", True)
            d.setdefault("tasks_affected", len(d.get("failing_task_ids", [])))
    return out
