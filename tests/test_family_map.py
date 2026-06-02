from studio.components.family_map import FamilyMap, init_map
from studio.components import meta_agent


def test_round_trip_through_text():
    fm = FamilyMap(works=["instructions: helps"], falsified=["tool_code: trap"])
    fm.add_pivot("try middleware for timeouts")
    again = FamilyMap.from_text(fm.to_text())
    assert again.works == fm.works
    assert again.falsified == fm.falsified
    assert again.pivot == fm.pivot


def test_empty_map_round_trips():
    fm = FamilyMap()
    assert FamilyMap.from_text(fm.to_text()) == fm


def test_promote_and_falsify_move_families():
    fm = FamilyMap(open=["middleware"])
    fm.promote("middleware", "fixed timeouts 4x")
    assert "middleware" in fm._family_names(fm.works)
    assert "middleware" not in fm._family_names(fm.open)
    fm.falsify("tool_code", "trap")
    assert fm.do_not_repeat() == ["tool_code"]


def test_falsify_removes_from_works():
    fm = FamilyMap(works=["tool_code: seemed to help"])
    fm.falsify("tool_code", "trap on deep audit")
    assert not fm._family_names(fm.works)
    assert "tool_code" in fm.do_not_repeat()


def test_rule_based_update_promotes_and_traps():
    fm = FamilyMap()
    meta_agent.rule_based_update(fm, accepted_families=["instructions", "tool_code"],
                                 traps=["tool_code"])
    assert "instructions" in fm._family_names(fm.works)
    assert "tool_code" in fm.do_not_repeat()
    assert "tool_code" not in fm._family_names(fm.works)  # trap not promoted


def test_init_map_writes_four_sections(tmp_path):
    p = tmp_path / "family_map.md"
    init_map(p)
    text = p.read_text()
    for title in ("Works (prefer)", "Falsified (do not repeat)", "Pivot toward", "Open / untried"):
        assert title in text
