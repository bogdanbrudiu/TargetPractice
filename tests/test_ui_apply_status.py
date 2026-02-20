from pathlib import Path


BASE = Path(__file__).resolve().parent.parent


def test_index_template_has_apply_status_elements_and_helpers():
    html = (BASE / "templates" / "index.html").read_text(encoding="utf-8")

    # Status placeholders (these are the visible UI parts)
    for elem_id in (
        "cameraApplyStatus",
        "targetApplyStatus",
        "weaponApplyStatus",
        "shootingApplyStatus",
    ):
        assert elem_id in html

    # Shared helpers (these are the behavior glue)
    assert "function _statusApplying" in html
    assert "function _statusApplied" in html
    assert "function _statusError" in html


def test_all_settings_actions_update_status_lines():
    html = (BASE / "templates" / "index.html").read_text(encoding="utf-8")

    # Camera settings
    assert "_statusApplying('cameraApplyStatus')" in html

    # Target settings
    assert "_statusApplying('targetApplyStatus')" in html

    # Weapon settings
    assert "_statusApplying('weaponApplyStatus')" in html

    # Shooting settings/actions
    assert "_statusApplying('shootingApplyStatus')" in html

    # Make sure we also surface errors for selection/actions, not only console.log
    for elem_id in (
        "cameraApplyStatus",
        "targetApplyStatus",
        "weaponApplyStatus",
        "shootingApplyStatus",
    ):
        assert f"_statusError('{elem_id}'" in html
