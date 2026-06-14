"""FailurePattern: a quantified, ranked failure cluster, and the ProposalBrief
the proposer receives for the targeted one.

Turns the diagnoser's per-cluster dicts into patterns with deterministic reach
and a **noise-aware expected win** (`max_gain = reach/|held_in|`; a pattern whose
best-case gain can't clear the noise floor is `unwinnable`). The proposer is then
aimed at the highest-expected-win pattern so it fixes the *class* that flips the
most tasks — not whatever single trace was in the prompt.

Research basis: prioritize by reach × fix-rate − regression (Errudite/RICE);
ground the proposal in the verifier's expected-vs-actual diffs (GEPA); a
class-level (not task-specific) edit is the overfit/reward-hacking guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailurePattern:
    pattern_id: str
    name: str
    blamed_part: str
    verifier_cause: str               # the failed-check signature (ground truth)
    task_ids: list[str]
    tasks_affected: int
    gt_diff_samples: list[str] = field(default_factory=list)
    addressable: bool = True
    confidence: float = 0.5
    max_gain: float = 0.0             # reach / |held_in| — the best-case gain
    expected_win: float = 0.0         # max_gain × confidence × addressable
    regression_risk: float = 0.0      # pass-set firing rate (0 until Phase 3 global anchoring)
    unwinnable: bool = False          # max_gain < noise_floor -> can't clear noise at this size


def aggregate(
    diagnosis: list[dict],
    *,
    held_in_size: int,
    noise_floor: float = 0.0,
    regression_risk: dict[str, float] | None = None,
) -> list[FailurePattern]:
    """Quantify + rank the diagnoser's clusters. Sorted addressable-first, then by
    ``expected_win − regression_risk`` (descending)."""
    n = max(1, held_in_size)
    rr = regression_risk or {}
    pats: list[FailurePattern] = []
    for d in diagnosis:
        ids = list(d.get("failing_task_ids", []))
        ta = int(d.get("tasks_affected") or len(ids))
        conf = float(d.get("confidence", 0.5) or 0.5)
        addr = bool(d.get("addressable", True))
        mg = ta / n
        pats.append(FailurePattern(
            pattern_id=str(d.get("pattern_id", "")),
            name=d.get("description") or d.get("root_cause", "") or d.get("verifier_cause", ""),
            blamed_part=d.get("blamed_part", "unclear"),
            verifier_cause=d.get("verifier_cause", ""),
            task_ids=ids,
            tasks_affected=ta,
            gt_diff_samples=list(d.get("gt_diff_samples", []))[:3],
            addressable=addr,
            confidence=conf,
            max_gain=mg,
            expected_win=mg * conf * (1.0 if addr else 0.0),
            regression_risk=float(rr.get(str(d.get("pattern_id", "")), 0.0)),
            unwinnable=mg < noise_floor,
        ))
    pats.sort(key=lambda p: (p.addressable, p.expected_win - p.regression_risk), reverse=True)
    return pats


def choose_target(
    patterns: list[FailurePattern], *, held_in_size: int, noise_floor: float = 0.0
) -> tuple[FailurePattern | None, bool]:
    """Pick the round's target: the top addressable pattern. If even it can't
    clear the noise floor, **bundle** all addressable failures into one synthetic
    target (bigger reach → bigger achievable gain). Returns (target, bundled).
    The target may still be ``unwinnable`` when the *whole* failure mass can't
    clear the floor — an honest signal that this batch size can't resolve a fix."""
    addr = [p for p in patterns if p.addressable]
    if not addr:
        return None, False
    top = addr[0]
    if not top.unwinnable:
        return top, False
    seen: set[str] = set()
    tasks: list[str] = []
    diffs: list[str] = []
    for p in addr:
        for t in p.task_ids:
            if t not in seen:
                seen.add(t)
                tasks.append(t)
        diffs.extend(p.gt_diff_samples)
    mg = len(tasks) / max(1, held_in_size)
    bundled = FailurePattern(
        pattern_id="bundle",
        name="bundled failures (no single pattern clears the noise floor)",
        blamed_part=top.blamed_part,
        verifier_cause="; ".join(sorted({p.verifier_cause for p in addr if p.verifier_cause})),
        task_ids=tasks, tasks_affected=len(tasks), gt_diff_samples=diffs[:3],
        addressable=True, confidence=top.confidence, max_gain=mg,
        expected_win=mg * top.confidence, unwinnable=mg < noise_floor,
    )
    return bundled, True


@dataclass
class ProposalBrief:
    """What the proposer receives for the targeted pattern — a class-level fix
    brief, not a raw trace."""

    pattern_name: str
    reach: int
    held_in_size: int
    expected_win: float
    blamed_part: str
    gt_diff_samples: list[str] = field(default_factory=list)
    do_not_break: list[str] = field(default_factory=list)
    unwinnable: bool = False
    bundled: bool = False

    @classmethod
    def from_pattern(cls, p: FailurePattern, *, held_in_size: int,
                     do_not_break: list[str] | None = None, bundled: bool = False) -> "ProposalBrief":
        return cls(
            pattern_name=p.name, reach=p.tasks_affected, held_in_size=held_in_size,
            expected_win=p.expected_win, blamed_part=p.blamed_part,
            gt_diff_samples=p.gt_diff_samples, do_not_break=do_not_break or [],
            unwinnable=p.unwinnable, bundled=bundled,
        )

    def render(self) -> str:
        lines = [
            f"TARGET FAILURE PATTERN: {self.pattern_name}",
            f"Reach: {self.reach} of {self.held_in_size} held-in failures share this mode"
            + (f" (expected win ~{self.expected_win:+.2f} of held-in)" if self.expected_win else ""),
            f"Blamed part: {self.blamed_part}",
        ]
        if self.gt_diff_samples:
            lines.append("Ground-truth diffs (verifier expected vs what happened):")
            lines += [f"  - {d}" for d in self.gt_diff_samples]
        if self.do_not_break:
            lines.append("Do NOT change behaviour on these currently-passing tasks: "
                         + ", ".join(self.do_not_break))
        lines.append("Propose ONE GENERAL rule/edit that fixes this whole CLASS of failures "
                     "(never a task-specific hack), and state how many of the affected tasks it "
                     "should flip.")
        if self.unwinnable:
            lines.append("(NOTE: at this batch size even a perfect fix may not clear the noise "
                         "floor — make the edit count.)")
        return "\n".join(lines)
