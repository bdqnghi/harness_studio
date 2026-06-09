"""Print the head-to-head verdict from the three result JSONs.

Reads /tmp/tb2_baseline.json, /tmp/tb2_ahe.json, /tmp/tb2_ours.json (override with
args) and reports pass-rate on the locked held-out pile for baseline / AHE / ours,
plus the win/lose verdict and (when available) cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str) -> dict | None:
    path = Path(p)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rate(d: dict | None) -> float | None:
    return None if d is None else d.get("pass_rate")


def main() -> None:
    baseline = load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tb2_baseline.json")
    ahe = load(sys.argv[2] if len(sys.argv) > 2 else "/tmp/tb2_ahe.json")
    ours = load(sys.argv[3] if len(sys.argv) > 3 else "/tmp/tb2_ours.json")

    print("=== Terminal-Bench 2 head-to-head — pass-rate on the locked held-out pile ===\n")
    rows = [("baseline (bare nexau)", baseline), ("AHE", ahe), ("harness_studio (ours)", ours)]
    for name, d in rows:
        r = rate(d)
        extra = ""
        if d and d.get("cost_per_point") is not None:
            extra = f"  (cost_per_point={d['cost_per_point']})"
        if d and d.get("per_task"):
            npass = sum(1 for v in d["per_task"].values() if v >= 1.0)
            extra += f"  [{npass}/{len(d['per_task'])} tasks]"
        print(f"  {name:24s}: {('%.3f' % r) if r is not None else 'PENDING':>7s}{extra}")

    br, ar, orr = rate(baseline), rate(ahe), rate(ours)
    print()
    if ar is not None and orr is not None:
        if orr > ar:
            print(f"  VERDICT: harness_studio WINS  (ours {orr:.3f} > AHE {ar:.3f}"
                  + (f", both over baseline {br:.3f}" if br is not None else "") + ")")
        elif orr == ar:
            print(f"  VERDICT: TIE on pass-rate ({orr:.3f}); compare cost_per_point / run more rounds.")
        else:
            print(f"  VERDICT: AHE leads ({ar:.3f} > ours {orr:.3f}) — tweak our method and re-run.")
    else:
        print("  VERDICT: pending — run both arms first.")


if __name__ == "__main__":
    main()
