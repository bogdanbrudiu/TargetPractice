from fastapi.testclient import TestClient
from pathlib import Path
from targetweb import server
from targetweb.targets import load_target
import uuid


def _build_test_client_with_tmp_config(tmp_path: Path) -> TestClient:
    # point the server to a temp config and reinitialize state
    cfg_path = tmp_path / "ha.ini"
    cfg_path.write_text("** Target settings **\n", encoding="utf-8")
    # bind a fresh AppState directly so endpoints have config_path
    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)
    server.state = server.AppState(latest_dir=tmp_path, target=tgt, caliber_mm=4.5, config_path=cfg_path)
    client = TestClient(server.app)
    return client


def test_list_targets_and_weapons():
    client = TestClient(server.app)
    r = client.get('/api/targets')
    assert r.status_code == 200
    assert 'targets' in r.json()
    r = client.get('/api/weapons')
    assert r.status_code == 200
    assert 'weapons' in r.json()


def test_select_target_and_persist(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    targets = client.get('/api/targets').json().get('targets', [])
    assert targets, 'no targets found in data/targets'
    name = targets[0]

    r = client.post('/api/targets/select', json={'name': name})
    assert r.status_code == 200
    assert r.json().get('active') == name

    # verify config via API and backup exists
    cfg_path = Path(tmp_path) / 'ha.ini'
    bak_path = cfg_path.with_suffix('.ini.bak')
    c = client.get('/api/config').json()
    assert c.get('TargetFilename') == name
    assert bak_path.exists()


def test_save_target_fields(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    payload = {
        "TargetFilename": "10m_air_pistol.tgt",
        "Scale": 0.5,
        "PositionX": -10,
        "PositionY": 20,
        "DistanceSimulated": 12.3,
        "DistanceReal": 6.7,
        "DistanceIsMetric": 1,
    }
    r = client.post('/api/config', json=payload)
    assert r.status_code == 200
    cfg = Path(tmp_path) / 'ha.ini'
    text = cfg.read_text(encoding='utf-8')
    for key, val in payload.items():
        assert str(val) in text


def test_select_weapon_and_save_fields(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    weapons = client.get('/api/weapons').json().get('weapons', [])
    assert weapons, 'no weapons found in data/weapons'
    name = weapons[0]
    r = client.post('/api/weapons/select', json={'name': name})
    assert r.status_code == 200
    assert r.json().get('active') == name

    payload = {
        "WeaponFilename": name,
        "Shooter": "TestShooter",
        "Rounds": 5,
        "Timer": 30,
        "TimeLimit": 60,
        "Countdown": 3,
        "HitLimit": 2,
        "SLH": 1,
        "Prepare": 4,
    }
    r = client.post('/api/config', json=payload)
    assert r.status_code == 200
    cfg = Path(tmp_path) / 'ha.ini'
    text = cfg.read_text(encoding='utf-8')
    for key, val in payload.items():
        assert str(val) in text


def test_weapon_file_is_editable_and_updates_state_caliber(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)

    base = Path(server.__file__).resolve().parent
    weapons_dir = (base.parent / "data" / "weapons").resolve()
    weapons_dir.mkdir(parents=True, exist_ok=True)

    weapon_filename = f"__pytest_weapon_{uuid.uuid4().hex}.gun"
    weapon_path = weapons_dir / weapon_filename

    try:
        weapon_path.write_text("Pytest Weapon\n4.50\nmm\n", encoding="utf-8")

        # Ensure it is listed
        weapons = client.get('/api/weapons').json().get('weapons', [])
        assert weapon_filename in weapons

        # Select it so updates should propagate to running state
        r = client.post('/api/weapons/select', json={'name': weapon_filename})
        assert r.status_code == 200
        assert r.json().get('active') == weapon_filename

        # Verify detail shows editable caliber in unit
        d = client.get(f'/api/weapons/{weapon_filename}').json()
        assert d.get('unit') == 'mm'
        assert float(d.get('caliber')) == 4.5

        # Update caliber (mm) and verify file updated
        r = client.post(f'/api/weapons/{weapon_filename}', json={'name': 'Pytest Weapon', 'caliber': 5.25, 'unit': 'mm'})
        assert r.status_code == 200
        assert r.json().get('ok') is True

        lines = [ln.strip() for ln in weapon_path.read_text(encoding='utf-8').splitlines()]
        assert lines[0] == 'Pytest Weapon'
        assert float(lines[1].replace(',', '.')) == 5.25
        assert lines[2].lower() == 'mm'

        # Since it is active, state caliber should update in mm
        assert server.state is not None
        assert abs(server.state.caliber_mm - 5.25) < 1e-6

        # Update caliber in inches and verify mm conversion updates
        r = client.post(f'/api/weapons/{weapon_filename}', json={'name': 'Pytest Weapon', 'caliber': 0.22, 'unit': 'inch'})
        assert r.status_code == 200
        assert r.json().get('ok') is True
        assert server.state is not None
        assert abs(server.state.caliber_mm - (0.22 * 25.4)) < 1e-6
    finally:
        try:
            weapon_path.unlink()
        except Exception:
            pass


def test_scale_only_persists_scale(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    # Initialize with some position
    init_payload = {"PositionX": -20, "PositionY": 15, "Scale": 0.4}
    r = client.post('/api/config', json=init_payload)
    assert r.status_code == 200
    # Scale up
    r = client.post('/api/target/scale', json={"factor": 1.1})
    assert r.status_code == 200
    cfg = client.get('/api/config').json()
    assert cfg["PositionX"] == -20
    assert cfg["PositionY"] == 15
    assert cfg["Scale"] is not None


def test_position_updates_only_position(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    init_payload = {"PositionX": 0, "PositionY": 0, "Scale": 0.5}
    r = client.post('/api/config', json=init_payload)
    assert r.status_code == 200
    # Nudge
    r = client.post('/api/target/position', json={"dx": 5, "dy": -3})
    assert r.status_code == 200
    cfg = client.get('/api/config').json()
    assert cfg["Scale"] == 0.5
    assert cfg["PositionX"] is not None
    assert cfg["PositionY"] is not None


def test_api_state_includes_fields(tmp_path):
    client = _build_test_client_with_tmp_config(tmp_path)
    # set config values
    payload = {
        "TargetFilename": "10m_air_pistol.tgt",
        "WeaponFilename": "default_weapon.gun",
        "Scale": 0.33,
        "PositionX": -12,
        "PositionY": 8,
        "Shooter": "UnitTester",
        "Rounds": 3,
        "Timer": 25,
        "TimeLimit": 60,
        "Countdown": 2,
        "HitLimit": 1,
        "SLH": 1,
        "Prepare": 4,
    }
    r = client.post('/api/config', json=payload)
    assert r.status_code == 200
    st = client.get('/api/state').json()
    assert 'Scale' in st and 'PositionX' in st and 'PositionY' in st
    assert 'Shooter' in st and 'Rounds' in st and 'Timer' in st
    assert 'TimeLimit' in st and 'Countdown' in st and 'HitLimit' in st and 'SLH' in st and 'Prepare' in st
    # values should match config posted (when baseline unknown, /api/state falls back to config)
    assert st['Scale'] == payload['Scale']
    assert st['PositionX'] == payload['PositionX']
    assert st['PositionY'] == payload['PositionY']
    assert st['Shooter'] == payload['Shooter']
