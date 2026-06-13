"""The hypothesis tree: the optimizer's persistent memory of what was tried.

Adapted from Arbor's Idea Tree (arXiv 2606.11926) to SHO's constraints: the
tree is a plain data structure owned by the deterministic orchestrator — no
LLM coordinator navigates it. Depth is fixed at two:

  * **direction** nodes group failures by signature (one per failure family);
  * **hypothesis** nodes are concrete, implementable edit ideas under a
    direction.

What it buys, in rollouts: a acceptance rejection becomes a durable lesson instead
of a counter. Falsified hypotheses are injected into future ideation as hard
"do not re-propose" constraints; ideas the *noise* killed stay re-proposable
(``rejected_noise``, retried at most twice) — Arbor never needed that
distinction because it ignores measurement noise; we cannot.

Selection is seeded Thompson sampling over per-direction Beta posteriors, so
the search explores at n=0 and concentrates as evidence accumulates, and the
whole thing replays deterministically from ``config.seed``.

The tree is saved atomically on every mutation: it must survive a crash
mid-campaign (it IS the resume state) and be readable mid-run (``tree.md``).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("pending", "tested_accepted", "rejected_noise", "falsified")

# A noise-killed hypothesis may be re-selected this many times before it is
# treated as exhausted (kept out of the frontier, but never a hard constraint).
MAX_NOISE_RETRIES = 2

# A rejected_noise child counts this much of a falsification in the posterior.
NOISE_WEIGHT = 0.5


@dataclass
class Node:
    id: str                      # "d3" (direction) / "d3h2" (hypothesis)
    parent_id: str | None
    kind: str                    # "direction" | "hypothesis"
    title: str
    mechanism: str = ""
    hypothesis: str = ""
    observable: str = ""         # the predicted measurable effect (checkable)
    status: str = "pending"
    evidence: dict = field(default_factory=dict)
    insight: str = ""            # <=200-word distilled lesson
    signature: dict = field(default_factory=dict)  # verifier_cause/agent_mechanism/addressable
    created_round: int = 0
    tested_round: int | None = None
    noise_rejections: int = 0


class IdeaTree:
    def __init__(self, path: Path, *, md_path: Path | None = None) -> None:
        self.path = Path(path)
        self.md_path = Path(md_path) if md_path else None
        self.nodes: dict[str, Node] = {}

    # --- construction ---

    @classmethod
    def load_or_create(cls, path: Path, *, md_path: Path | None = None) -> "IdeaTree":
        tree = cls(path, md_path=md_path)
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text())
            for raw in data.get("nodes", []):
                node = Node(**raw)
                tree.nodes[node.id] = node
        return tree

    def add_direction(
        self, title: str, mechanism: str, signature: dict, round_idx: int
    ) -> Node:
        nid = f"d{sum(1 for n in self.nodes.values() if n.kind == 'direction') + 1}"
        node = Node(
            id=nid, parent_id=None, kind="direction", title=title,
            mechanism=mechanism, signature=dict(signature or {}),
            created_round=round_idx,
        )
        self.nodes[nid] = node
        self.save()
        return node

    def add_hypothesis(
        self, direction_id: str, *, title: str, mechanism: str,
        hypothesis: str, observable: str, round_idx: int,
    ) -> Node:
        parent = self.nodes[direction_id]
        if parent.kind != "direction":
            raise ValueError(f"{direction_id} is not a direction node")
        nid = f"{direction_id}h{len(self.children(direction_id)) + 1}"
        node = Node(
            id=nid, parent_id=direction_id, kind="hypothesis", title=title,
            mechanism=mechanism, hypothesis=hypothesis, observable=observable,
            created_round=round_idx,
        )
        self.nodes[nid] = node
        self.save()
        return node

    # --- mutation ---

    def set_status(
        self, node_id: str, status: str, *, evidence: dict | None = None,
        tested_round: int | None = None,
    ) -> Node:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        node = self.nodes[node_id]
        node.status = status
        if evidence is not None:
            node.evidence = dict(evidence)
        if tested_round is not None:
            node.tested_round = tested_round
        self.save()
        return node

    def mark_noise_retry(self, node_id: str) -> Node:
        node = self.nodes[node_id]
        node.noise_rejections += 1
        self.save()
        return node

    def set_insight(self, node_id: str, text: str) -> Node:
        node = self.nodes[node_id]
        node.insight = _truncate_words(text, 200)
        self.save()
        return node

    # --- queries ---

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def directions(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "direction"]

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def frontier(self, direction_id: str) -> list[Node]:
        """Hypotheses ready to test: pending first (FIFO), then noise-killed
        ones that still have retries left. Falsified ideas never reappear."""
        kids = self.children(direction_id)
        pending = [n for n in kids if n.status == "pending"]
        retryable = [
            n for n in kids
            if n.status == "rejected_noise" and n.noise_rejections < MAX_NOISE_RETRIES
        ]
        return pending + retryable

    def pending_titles(self) -> list[str]:
        return [n.title for n in self.nodes.values()
                if n.kind == "hypothesis" and n.status == "pending"]

    def falsified_constraints(self) -> list[str]:
        out = []
        for n in self.nodes.values():
            if n.kind == "hypothesis" and n.status == "falsified":
                line = f"{n.title}: {n.hypothesis}"
                if n.insight:
                    line += f" — {n.insight}"
                out.append(line)
        return out

    def validated_insights(self, direction_id: str) -> list[str]:
        """Lessons a new hypothesis under this direction should inherit: the
        direction's own summary plus every tested sibling's insight."""
        direction = self.nodes[direction_id]
        out = [direction.insight] if direction.insight else []
        for n in self.children(direction_id):
            if n.insight and n.status != "pending":
                out.append(n.insight)
        return out

    # --- selection ---

    def posterior(self, direction_id: str) -> tuple[float, float]:
        kids = self.children(direction_id)
        accepted = sum(1 for n in kids if n.status == "tested_accepted")
        falsified = sum(1 for n in kids if n.status == "falsified")
        noise = sum(1 for n in kids if n.status == "rejected_noise")
        return 1.0 + accepted, 1.0 + falsified + NOISE_WEIGHT * noise

    def select_direction(self, rng: random.Random) -> Node | None:
        """Seeded Thompson sampling: draw from each selectable direction's Beta
        posterior, take the argmax. Falsified directions are never selected."""
        candidates = [d for d in self.directions() if d.status != "falsified"]
        if not candidates:
            return None
        best, best_draw = None, -1.0
        for d in sorted(candidates, key=lambda n: n.id):  # stable order
            alpha, beta = self.posterior(d.id)
            draw = rng.betavariate(alpha, beta)
            if draw > best_draw:
                best, best_draw = d, draw
        return best

    # --- persistence ---

    def save(self) -> None:
        payload = {"nodes": [asdict(n) for n in self.nodes.values()]}
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, self.path)  # atomic: a crash never leaves a torn tree
        if self.md_path:
            try:
                self.md_path.write_text(self.to_markdown())
            except OSError:
                pass

    def to_markdown(self) -> str:
        lines = ["# Hypothesis tree", ""]
        for d in sorted(self.directions(), key=lambda n: n.id):
            alpha, beta = self.posterior(d.id)
            lines.append(f"## {d.id} {d.title}  [{d.status}]  Beta({alpha:g},{beta:g})")
            if d.mechanism:
                lines.append(f"- mechanism: {d.mechanism}")
            if d.insight:
                lines.append(f"- insight: {d.insight}")
            for h in sorted(self.children(d.id), key=lambda n: n.id):
                mark = {"pending": " ", "tested_accepted": "+",
                        "rejected_noise": "~", "falsified": "x"}[h.status]
                lines.append(f"- [{mark}] {h.id} {h.title} ({h.status})")
                if h.insight:
                    lines.append(f"    - {h.insight}")
            lines.append("")
        return "\n".join(lines)


def classify_rejection(decision, noise_floor: float) -> str:
    """Was a acceptance rejection a *falsification* (clear regression beyond noise)
    or just an unresolved result inside the noise band?

    ``decision.regressed`` alone is not enough: the borderline-resolution path
    also sets it for tiny in-noise negatives. We require the worst split's gain
    to clear the residual noise threshold for the number of runs actually used.
    Only falsifications become hard constraints; noise rejections stay
    re-proposable — hard-constraining them would let noise permanently kill
    good ideas.
    """
    threshold = max(0.0, noise_floor) / math.sqrt(max(1, decision.runs_used))
    worst = min(decision.gain, decision.regression_gain)
    if decision.regressed and worst < -threshold:
        return "falsified"
    return "rejected_noise"


def mutation_event(node: Node, change: str) -> dict:
    """Payload for a ``tree_mutation`` progress event."""
    return {
        "node": node.id, "kind": node.kind, "title": node.title,
        "status": node.status, "change": change,
    }


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit])
