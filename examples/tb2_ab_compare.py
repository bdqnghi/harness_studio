#!/usr/bin/env python
"""Compare the two arms of a classic-vs-tree A/B cell.

Reads each arm's workspace artifacts (``lift.json``, ``evidence.jsonl``,
``progress.jsonl``, and the tree arm's ``idea_tree.json``) and emits the
pre-registered readout:

* PRIMARY — per-arm locked-test lift ± SE, and the per-task
  difference-of-differences (both arms share the same baseline scores via the
  score cache, so ``d_t = opt_tree(t) − opt_classic(t)``) with a 95% CI
  against the detectable floor.
* SECONDARY — the efficiency metrics that decide a within-noise primary:
  accepts, fresh rollouts, rollouts per accepted edit, Tier-A implementation
  runs per accepted edit, gate accept rates, wall-clock, falsified/constraint
  counts.

  python examples/tb2_ab_compare.py \
      --classic artifacts/ab_nexau/classic --tree artifacts/ab_nexau/tree \
      --out artifacts/ab_nexau/report
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def arm_metrics(ws: Path) -> dict:
    lift = json.loads((ws / "lift.json").read_text()) if (ws / "lift.json").exists() else {}
    evidence = _read_jsonl(ws / "evidence.jsonl")
    progress = _read_jsonl(ws / "progress.jsonl")

    rounds = [e for e in evidence if "accepted" in e]
    accepts = sum(1 for e in rounds if e["accepted"])
    gates = [e for e in progress if e.get("event") == "gate_decision"]
    round_ends = [e for e in progress if e.get("event") == "round_end"]
    proposals = [e for e in progress if e.get("event") == "proposal_done"]
    setup = next((e for e in progress if e.get("event") == "setup_done"), {})

    task_runs = round_ends[-1].get("task_runs", 0) if round_ends else 0
    tier_a_runs = sum(len(e.get("strategies", [])) for e in proposals)
    out = {
        "workspace": str(ws),
        "lift": lift.get("lift"), "se": lift.get("se"),
        "detectable_final": lift.get("detectable_final"),
        "per_task_lift": lift.get("per_task_lift", {}),
        "rounds": len(rounds), "accepts": accepts,
        "accept_rate_per_round": round(accepts / len(rounds), 3) if rounds else None,
        "gate_attempts": len(gates),
        "accept_rate_per_attempt": (
            round(sum(1 for g in gates if g.get("accept")) / len(gates), 3) if gates else None
        ),
        "fresh_task_runs": task_runs,
        "task_runs_per_accept": round(task_runs / accepts, 1) if accepts else None,
        "tier_a_runs": tier_a_runs,
        "tier_a_per_accept": round(tier_a_runs / accepts, 2) if accepts else None,
        "wall_sec_total": round(sum(e.get("wall_sec", 0.0) for e in round_ends), 1),
        "wobble": setup.get("wobble"),
        "judging_trajectory": [e.get("new_score") for e in rounds],
    }
    tree_file = ws / "idea_tree.json"
    if tree_file.exists():
        nodes = json.loads(tree_file.read_text()).get("nodes", [])
        hyps = [n for n in nodes if n.get("kind") == "hypothesis"]
        out["tree"] = {
            "directions": sum(1 for n in nodes if n.get("kind") == "direction"),
            "hypotheses": len(hyps),
            "falsified": sum(1 for n in hyps if n.get("status") == "falsified"),
            "rejected_noise": sum(1 for n in hyps if n.get("status") == "rejected_noise"),
            "accepted": sum(1 for n in hyps if n.get("status") == "tested_accepted"),
            "pending": sum(1 for n in hyps if n.get("status") == "pending"),
        }
    return out


def difference_of_differences(classic: dict, tree: dict) -> dict:
    ct, tt = classic["per_task_lift"], tree["per_task_lift"]
    shared = sorted(set(ct) & set(tt))
    if not shared:
        return {"n": 0, "verdict": "no shared locked-test tasks — arms not comparable"}
    d = {t: tt[t] - ct[t] for t in shared}
    mean = statistics.mean(d.values())
    sd = statistics.stdev(d.values()) if len(d) > 1 else 0.0
    se = sd / (len(d) ** 0.5)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    floor = max(classic.get("detectable_final") or 0.0, tree.get("detectable_final") or 0.0)
    if lo > 0:
        verdict = "TREE WINS on the primary (95% CI excludes 0)"
    elif hi < 0:
        verdict = "TREE LOSES on the primary (95% CI excludes 0)"
    else:
        verdict = (
            f"WITHIN NOISE (|DoD| < floor {floor:.3f} at n={len(d)}): no detectable "
            "quality difference — the verdict falls to the efficiency secondaries"
        )
    return {"n": len(d), "mean": round(mean, 4), "se": round(se, 4),
            "ci95": [round(lo, 4), round(hi, 4)], "floor": round(floor, 4),
            "per_task": {t: round(v, 4) for t, v in d.items()}, "verdict": verdict}


def _fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, (int, float)) else "?"


def render(classic: dict, tree: dict, dod: dict) -> str:
    lines = ["# A/B report: classic vs tree", ""]
    lines.append("## Primary (locked test)")
    for name, m in (("classic", classic), ("tree", tree)):
        lines.append(f"- {name}: lift {_fmt(m['lift'], '+.3f')} ± {_fmt(m['se'])} "
                     f"(detectable {_fmt(m['detectable_final'])})")
    if dod.get("n"):
        lines.append(f"- DoD (tree − classic): {dod['mean']:+.4f} ± {dod['se']:.4f}, "
                     f"95% CI [{dod['ci95'][0]:+.4f}, {dod['ci95'][1]:+.4f}], "
                     f"floor {dod['floor']:.3f}")
    lines.append(f"- **{dod.get('verdict', '?')}**")
    lines += ["", "## Secondary (efficiency)", ""]
    rows = [
        ("rounds", "rounds"), ("accepts", "accepts"),
        ("accept rate / round", "accept_rate_per_round"),
        ("gate attempts", "gate_attempts"),
        ("accept rate / attempt", "accept_rate_per_attempt"),
        ("fresh task runs", "fresh_task_runs"),
        ("task runs / accept", "task_runs_per_accept"),
        ("Tier-A runs", "tier_a_runs"),
        ("Tier-A / accept", "tier_a_per_accept"),
        ("wall sec (rounds)", "wall_sec_total"),
        ("wobble", "wobble"),
    ]
    lines.append(f"| metric | classic | tree |")
    lines.append("|---|---|---|")
    for label, key in rows:
        lines.append(f"| {label} | {classic.get(key)} | {tree.get(key)} |")
    if tree.get("tree"):
        t = tree["tree"]
        lines += ["", f"Tree: {t['directions']} directions, {t['hypotheses']} hypotheses "
                      f"({t['accepted']} accepted, {t['falsified']} falsified, "
                      f"{t['rejected_noise']} noise-rejected, {t['pending']} pending)"]
    lines += ["", "## Judging trajectory",
              f"- classic: {classic['judging_trajectory']}",
              f"- tree:    {tree['judging_trajectory']}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic", required=True, help="classic arm workspace")
    ap.add_argument("--tree", required=True, help="tree arm workspace")
    ap.add_argument("--out", default=None, help="report dir (default: print only)")
    args = ap.parse_args()

    classic = arm_metrics(Path(args.classic))
    tree = arm_metrics(Path(args.tree))
    dod = difference_of_differences(classic, tree)
    text = render(classic, tree, dod)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(
            {"classic": classic, "tree": tree, "difference_of_differences": dod},
            indent=2))
        (out / "report.md").write_text(text + "\n")
        print(f"\nreport -> {out / 'report.json'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
