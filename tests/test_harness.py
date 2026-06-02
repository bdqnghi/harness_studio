from studio.benchmark.toy import build_toy_harness
from studio.benchmark import toy_fixes


def test_files_and_read(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    files = h.files()
    assert "tools.py" in files and "instructions.txt" in files
    assert "ENABLE echo" in h.read_file("instructions.txt")


def test_copy_is_independent(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    c = h.copy_to(tmp_path / "c")
    toy_fixes.enable_upper(c.root)
    # editing the copy must not change the original
    assert "ENABLE upper" in c.read_file("instructions.txt")
    assert "ENABLE upper" not in h.read_file("instructions.txt")


def test_content_hash_changes_on_edit(tmp_path):
    h = build_toy_harness(tmp_path / "h")
    before = h.content_hash()
    toy_fixes.fix_reverse(h.root)
    assert h.content_hash() != before
