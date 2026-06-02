"""Meta-agent (PRD §5.10): revise the search rules so it escapes plateaus.

Two speeds of map maintenance (PRD §8, §11 Q5):

* ``rule_based_update`` — the cheap default, run every segment boundary in code:
  promote families confirmed to help, record deep-audit traps as falsified.
* ``escalate`` — the Tier-A coding agent, triggered only on an observed plateau.
  It reads the segment's evidence on disk and makes ONE mechanism edit to the
  family map (a pivot directive). Its skill forbids touching the evaluator,
  candidates, or scores (the AEVO protection rule).

The meta-agent never proposes strategies and never touches the gate.
"""

from __future__ import annotations

from pathlib import Path

from ..backends.base import AgentResult, Backend
from .family_map import FamilyMap

TAG = "meta"
SKILL_PATH = Path(__file__).resolve().parent.parent / "skills" / "meta_agent" / "SKILL.md"


def load_skill() -> str:
    return SKILL_PATH.read_text()


def rule_based_update(
    fmap: FamilyMap, accepted_families: list[str], traps: list[str]
) -> None:
    """Bookkeeping the segment proved on its own — no deliberation needed."""
    for fam in traps:
        fmap.falsify(fam, "passed the fast gate but regressed on the deep audit (trap)")
    for fam in accepted_families:
        if fam not in traps:  # a trap is not also a 'works'
            fmap.promote(fam, "improved the harness and held up on the deep audit")


def escalate(
    backend: Backend,
    mechanism_dir: Path,
    *,
    model: str | None = None,
) -> AgentResult:
    """Run the Tier-A meta-agent on the mechanism directory (family_map.md +
    segment_evidence.md). It makes one edit to the family map."""
    return backend.run_agent(
        "A plateau was detected this segment (little or no accepted improvement). "
        "Read segment_evidence.md and make exactly ONE edit to family_map.md — most "
        "likely a 'Pivot toward' directive — to redirect the next segment's search.",
        workspace=mechanism_dir,
        skill=load_skill(),
        tag=TAG,
        model=model,
    )
