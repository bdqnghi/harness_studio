from pathlib import Path

from studio.backends.mock import MockBackend
from studio.benchmark.toy import ToyBenchmark, build_toy_harness
from studio.benchmark import toy_fixes
from studio.components import structural_check


def test_passes_for_bootable_candidate(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    res = structural_check.check(h, ToyBenchmark(), backend=None)
    assert res.ok and not res.repaired


def test_catches_syntax_error(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    toy_fixes.break_boot(h.root)
    res = structural_check.check(h, ToyBenchmark(), backend=None, allow_repair=False)
    assert not res.ok and "tools.py" in res.error


def test_repair_attempt_fixes_it(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    toy_fixes.break_boot(h.root)

    def repair_action(root: Path) -> None:
        # the "repair agent" rewrites tools.py back to something bootable
        (root / "tools.py").write_text("def _e(a):\n    return a\n\n\nOPS = {'echo': _e}\n")

    backend = MockBackend(agent_actions={"strategist-repair": [repair_action]})
    res = structural_check.check(h, ToyBenchmark(), backend=backend, allow_repair=True)
    assert res.ok and res.repaired


def test_repair_failure_still_reports(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    toy_fixes.break_boot(h.root)
    backend = MockBackend(agent_actions={"strategist-repair": [toy_fixes.noop]})
    res = structural_check.check(h, ToyBenchmark(), backend=backend, allow_repair=True)
    assert not res.ok and res.repaired  # tried to repair, still broken
