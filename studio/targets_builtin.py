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
    from .benchmark.tau2 import (
        AGENT_INSTRUCTION_FILE, instruction_injectable, policy_files_for,
    )

    # The contract the runtime executes: the agent reads its operating policy
    # (and, when the source supports it, a behavioral instruction) from these
    # file(s). The coding agent generates them so the harness is runnable.
    files = list(policy_files_for(domain))
    if instruction_injectable():
        files.append(AGENT_INSTRUCTION_FILE)
    runner_contract = (
        "The runtime runs a customer-service agent that follows an operating policy "
        f"read from these file(s): {', '.join(files)}. Write the policy as markdown prose "
        "rules the agent must obey (identity verification before acting on an account, and "
        "the domain's action/eligibility rules). "
        + (f"Also write {AGENT_INSTRUCTION_FILE}: a short behavioral instruction (e.g. each "
           "turn either message the user OR make a tool call, not both)."
           if AGENT_INSTRUCTION_FILE in files else "")
    )
    return ColdStartBrief(
        domain=f"{domain} customer-service tool-use dialogue",
        io_contract=(
            "Multi-turn dialogue with a customer. Resolve their request by calling "
            "the domain tools. The environment grades the final database state plus "
            "whether the required actions were taken — so you must actually execute "
            "the right tool calls, not just describe them."
        ),
        tools=_TAU2_TOOLS.get(domain, []),
        runner_contract=runner_contract,
        extra_notes="Follow the domain rules strictly; verify identity before acting on an account.",
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
