"""Built-in Target registrations — import this once to populate the registry.

Adding a benchmark to SHO is now: implement a thin ``Benchmark`` adapter, then
register a ``Target`` here. The optimizer/driver resolve everything through
``targets.get_target(name)`` and never learn benchmark specifics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .targets import ColdStartBrief, Target, TargetConfig, ToolSpec, register


# --- tau2-bench (warm: climb the shipped policy; cold: synthesize a bare one) --

# A representative tool set per domain for the cold-start brief (the bare policy
# only references these for context; tau2 supplies the full toolset at runtime).
_TAU2_TOOLS = {
    "airline": [
        ToolSpec("get_user_details", "get_user_details(user_id)", "look up a user + their reservations"),
        ToolSpec("get_reservation_details", "get_reservation_details(id)", "look up a reservation"),
        ToolSpec("book_reservation", "book_reservation(...)", "book a new reservation"),
        ToolSpec("update_reservation", "update_reservation(...)", "change flights/baggage on a reservation"),
        ToolSpec("cancel_reservation", "cancel_reservation(id)", "cancel a reservation"),
        ToolSpec("get_flight_status", "get_flight_status(...)", "check a flight's status"),
        ToolSpec("transfer_to_human", "transfer_to_human()", "escalate to a human agent"),
    ],
    "retail": [
        ToolSpec("get_user_details", "get_user_details(user_id)", "look up a user + their orders"),
        ToolSpec("get_order_details", "get_order_details(order_id)", "look up an order"),
        ToolSpec("cancel_pending_order", "cancel_pending_order(...)", "cancel a pending order"),
        ToolSpec("modify_pending_order", "modify_pending_order(...)", "modify a pending order"),
        ToolSpec("return_delivered_order", "return_delivered_order(...)", "process a return"),
        ToolSpec("exchange_delivered_order", "exchange_delivered_order(...)", "process an exchange"),
        ToolSpec("transfer_to_human", "transfer_to_human()", "escalate to a human agent"),
    ],
}


def _tau2_brief(domain: str) -> ColdStartBrief:
    from .benchmark.tau2 import AGENT_INSTRUCTION_FILE, instruction_injectable

    # When tau2's source can consume a mutated agent instruction, also cold-seed
    # a deliberately bare one (alongside the bare policy) so both levers have
    # headroom and the cold harness matches the expanded editable surface.
    extra = {}
    if instruction_injectable():
        extra[AGENT_INSTRUCTION_FILE] = (
            "You are a customer-service agent. In each turn either send the user "
            "a message OR make a tool call (not both). Use the tools to act.\n"
        )
    return ColdStartBrief(
        domain=f"{domain} customer-service tool-use dialogue (tau2-bench)",
        io_contract=(
            "Multi-turn dialogue with a customer. Resolve their request by calling "
            "the domain tools. The environment grades the final database state plus "
            "whether the required actions were taken — so you must actually execute "
            "the right tool calls, not just describe them."
        ),
        tools=_TAU2_TOOLS.get(domain, []),
        template="policy",
        extra_notes="Follow the domain rules strictly; verify identity before acting on an account.",
        extra_files=extra,
    )


def _tau2_target(domain: str, baseline: float) -> Target:
    from .benchmark.tau2 import (
        Tau2Benchmark, tau2_part_map, tau2_seed_harness,
    )

    def make_bench(cfg: TargetConfig):
        return Tau2Benchmark(
            domain=domain,
            model=cfg.model,
            user_model=cfg.extra.get("user_model", "gpt-4.1-mini"),
            k=cfg.k,
            n_concurrent=cfg.n_concurrent,
            real=cfg.real,
        )

    def seed():
        d = Path(tempfile.mkdtemp(prefix=f"tau2-seed-{domain}-"))
        return tau2_seed_harness(domain, d)

    # Cold start only for the single-prose-policy domains (airline/retail use
    # policy.md, which the "policy" cold template writes; telecom splits its
    # policy across files, so warm-only for now).
    cold = (lambda d=domain: _tau2_brief(d)) if domain in ("airline", "retail") else None

    return Target(
        name=f"tau2-{domain}",
        make_benchmark=make_bench,
        part_map=lambda d=domain: tau2_part_map(d),
        seed_harness=seed,
        cold_start_brief=cold,
        baseline_score=baseline,
        baseline_note=f"tau2 {domain} Pass^1 (gpt-4.1, sierra paper)",
    )


for _dom, _bl in (("airline", 0.56), ("retail", 0.74), ("telecom", 0.34)):
    register(f"tau2-{_dom}", (lambda d=_dom, b=_bl: _tau2_target(d, b)))
