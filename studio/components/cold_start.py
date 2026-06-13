"""Cold start: synthesize a runnable agent harness when a benchmark ships none.

This is what lets SHO hill-climb on a benchmark with NO baseline harness (e.g.
BrowseComp): instead of an input harness, the benchmark provides a
``ColdStartBrief`` (domain + I/O contract + available tools + template), and we
synthesize a minimal, runnable-but-unoptimized harness from it. The optimizer
then treats that as round 0 and climbs.

Design:
* The template skeleton is laid down **deterministically** (built-in files), so
  the harness layout and the editable surface are stable and the synthesis is
  testable with a MockBackend.
* A single Tier-B ``prompt_json`` call fills the *content* (initial system
  prompt + per-tool usage notes + loop guidance) from the brief. That filled
  content is written into the template.
* The result is a harness whose 7-part surface is exactly what the optimizer
  knows how to mutate (see :func:`cold_start_part_map`).

The caller (driver) is responsible for the *validation* step — boot_check + a
1-task smoke run via the actual benchmark, with bounded retry — because only the
benchmark can run a task. Keeping that out of here leaves this module
benchmark-independent and unit-testable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import schemas
from ..harness import Harness
from ..parts import PartMap, PartType

SYNTH_TAG = "cold-start"

# --- built-in templates ----------------------------------------------------
#
# Each template is a dict of {relative_path: skeleton_text}. Skeletons contain
# {{PLACEHOLDERS}} filled from the synthesis call. The file split matches the
# 7-part PartMap so the optimizer can target each part independently.

_REACT_AGENT_PY = '''\
"""Minimal ReAct agent loop (editable MIDDLEWARE / control flow + MEMORY).

The benchmark runner imports ``run_episode`` and supplies ``call_model`` (the
LLM) and ``tools`` (the real tool implementations for this benchmark). This file
owns the agent's control flow: how it is prompted, how tool results are folded
back into history, how many steps it takes, and when it stops.
"""
import json
from pathlib import Path

MAX_STEPS = {{MAX_STEPS}}


def load_text(name):
    return (Path(__file__).parent / name).read_text()


def run_episode(task_prompt, call_model, tools):
    """Drive one task. ``call_model(messages, tool_schemas) -> message`` and
    ``tools`` is {name: callable}. Returns the final answer string."""
    system = load_text("system_prompt.md") + "\\n\\n" + load_text("tools.md")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task_prompt}]
    tool_schemas = json.loads(load_text("tool_schemas.json"))
    for _ in range(MAX_STEPS):
        msg = call_model(messages, tool_schemas)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "")
        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            result = tools[name](**args) if name in tools else f"unknown tool {name}"
            if name == "finish":
                return args.get("answer", str(result))
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": str(result)})
    return ""  # ran out of steps without finishing
'''

_TEMPLATES: dict[str, dict[str, str]] = {
    "react": {
        "system_prompt.md": "{{SYSTEM_PROMPT}}\n",
        "tools.md": "{{TOOLS_MD}}\n",
        "tool_schemas.json": "{{TOOL_SCHEMAS}}\n",
        "agent.py": _REACT_AGENT_PY,
        "config.json": '{\n  "max_steps": {{MAX_STEPS}},\n  "template": "react"\n}\n',
    },
    # "policy": a single prose policy file (e.g. tau2's harness). Cold start
    # writes a deliberately MINIMAL policy so there is guaranteed headroom for
    # SHO to climb — it discovers and adds the domain-specific rules from
    # observed failures, rather than starting from a hand-tuned baseline.
    "policy": {"policy.md": "{{POLICY}}\n"},
}

_DEFAULT_MAX_STEPS = 12

# Filename the "policy" template writes (matches single-prose-file adapters).
POLICY_FILE = "policy.md"


def cold_start_part_map(template: str = "react") -> PartMap:
    """The editable surface of a cold-started harness (stable per template)."""
    if template == "react":
        return PartMap(
            parts={
                PartType.INSTRUCTIONS: ["system_prompt.md"],
                PartType.TOOL_DESCRIPTIONS: ["tools.md", "tool_schemas.json"],
                PartType.MIDDLEWARE: ["agent.py"],
                PartType.MEMORY: ["agent.py"],   # history handling lives in the loop
            },
            do_not_touch=[],
        )
    if template == "policy":
        return PartMap(parts={PartType.INSTRUCTIONS: [POLICY_FILE]}, do_not_touch=[])
    raise ValueError(f"unknown cold-start template {template!r}")


def _minimal_policy(brief) -> str:
    """A deterministic, deliberately bare policy — the cold-start seed for a
    prose-policy harness. Generic on purpose: it states the role, the I/O
    contract, and the available tools, but NONE of the domain-specific rules
    (refund eligibility, confirmation steps, ...). SHO must discover those, so
    the headroom is real and the climb is measurable."""
    tools = "\n".join(f"- {t.name}: {t.doc}" for t in brief.tools) or "- (use the tools provided by the environment)"
    return (
        f"# Policy\n\n"
        f"You are an assistant for: {brief.domain}.\n\n"
        f"{brief.io_contract}\n\n"
        f"## Available tools\n{tools}\n\n"
        f"## Conduct\n"
        f"- Be helpful and resolve the user's request.\n"
        f"- Use the available tools to take actions rather than guessing.\n"
        f"- Confirm with the user before any irreversible action.\n"
        + (f"\n{brief.extra_notes}\n" if brief.extra_notes else "")
    )


def _tool_schemas(brief) -> list[dict]:
    """OpenAI-style function schemas the model sees, from the brief's tools."""
    out = []
    for t in brief.tools:
        out.append({"name": t.name, "signature": t.signature, "description": t.doc})
    return out


def _synthesize(backend, brief, *, model=None) -> dict:
    """One Tier-B call: initial system prompt + per-tool notes + loop guidance."""
    tools = "\n".join(f"- {t.name}{_paren(t.signature)}: {t.doc}" for t in brief.tools) or "(none)"
    prompt = (
        "You are bootstrapping the INITIAL system prompt for an autonomous agent "
        "that will be iteratively improved later. Make it correct and functional, "
        "not clever — a solid baseline.\n\n"
        f"Domain: {brief.domain}\n"
        f"Task I/O contract: {brief.io_contract}\n"
        f"Available tools:\n{tools}\n"
        + (f"Notes: {brief.extra_notes}\n" if brief.extra_notes else "")
        + "\nReturn JSON: system_prompt (the agent's system instructions — tell it "
        "to use the tools and to call finish(answer) when done), tool_notes (one "
        "short usage note per tool), loop_guidance (optional: how to plan, iterate, "
        "and decide when to stop)."
    )
    return backend.prompt_json(prompt, schemas.COLD_START, tag=SYNTH_TAG, model=model)


def _paren(sig: str) -> str:
    """Render just the arg list of a signature for the prompt listing."""
    if "(" in sig:
        return "(" + sig.split("(", 1)[1].split(")")[0] + ")"
    return ""


def bootstrap_harness(backend, brief, workdir: Path, *, model=None) -> Harness:
    """Synthesize a runnable round-0 harness from a ColdStartBrief.

    Lays down the template, fills it from one synthesis call, returns the
    Harness. Does NOT run the benchmark — the driver validates (boot_check +
    smoke) and may call this again on failure."""
    template = brief.template
    if template not in _TEMPLATES:
        raise ValueError(f"unknown cold-start template {template!r}")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    harness = Harness(workdir)

    # The "policy" template is a single deterministic prose file — no LLM call
    # needed (and a bare seed is exactly what we want for guaranteed headroom).
    if template == "policy":
        harness.write_file(POLICY_FILE, _minimal_policy(brief) + "\n")
        for rel, content in (brief.extra_files or {}).items():
            harness.write_file(rel, content)
        return harness

    synth = _synthesize(backend, brief, model=model)
    system_prompt = str(synth.get("system_prompt", "")).strip()
    notes = {n["name"]: n["note"] for n in synth.get("tool_notes", []) if isinstance(n, dict)}
    guidance = str(synth.get("loop_guidance", "")).strip()

    tools_md_lines = ["# Tools", ""]
    for t in brief.tools:
        tools_md_lines.append(f"## {t.name}{_paren(t.signature)}")
        tools_md_lines.append(t.doc)
        if t.name in notes:
            tools_md_lines.append(f"Usage: {notes[t.name]}")
        tools_md_lines.append("")
    if guidance:
        tools_md_lines += ["# How to work", guidance, ""]
    tools_md = "\n".join(tools_md_lines)

    subst = {
        "SYSTEM_PROMPT": system_prompt or f"You are an agent for: {brief.domain}.",
        "TOOLS_MD": tools_md,
        "TOOL_SCHEMAS": json.dumps(_tool_schemas(brief), indent=2),
        "MAX_STEPS": str(_DEFAULT_MAX_STEPS),
    }
    for rel, skeleton in _TEMPLATES[template].items():
        text = skeleton
        for k, v in subst.items():
            text = text.replace("{{" + k + "}}", v)
        harness.write_file(rel, text)
    for rel, content in (brief.extra_files or {}).items():
        harness.write_file(rel, content)
    return harness
