from pathlib import Path

from fastapi.testclient import TestClient

from targetweb import server
from targetweb.targets import load_target


def _build_client(tmp_path: Path, ini_text: str) -> TestClient:
    cfg_path = tmp_path / "ha.ini"
    cfg_path.write_text(ini_text, encoding="utf-8")

    base = Path(server.__file__).resolve().parent
    default_target = (base.parent / "data" / "targets" / "10m_air_pistol.tgt").resolve()
    tgt = load_target(default_target)
    server.state = server.AppState(latest_dir=tmp_path, target=tgt, caliber_mm=4.5, config_path=cfg_path)
    return TestClient(server.app)


def test_prepare_zero_starts_immediately(tmp_path):
    client = _build_client(
        tmp_path,
        "\n".join(
            [
                "** Shooting settings **",
                "Prepare = 0",
                "Rounds = 10",
                "HitLimit = 0",
                "",
            ]
        ),
    )

    r = client.post("/api/shooting/start")
    assert r.status_code == 200
    body = r.json()
    assert body.get("shooting_status") == "running"
    assert int(body.get("shooting_prepare_remaining", 0)) == 0


def test_rounds_auto_stop_even_if_hitlimit_disabled(tmp_path):
    client = _build_client(
        tmp_path,
        "\n".join(
            [
                "** Shooting settings **",
                "Prepare = 0",
                "Rounds = 2",
                "HitLimit = 0",
                "",
            ]
        ),
    )

    r = client.post("/api/shooting/start")
    assert r.status_code == 200
    assert r.json().get("shooting_status") == "running"

    assert server.state is not None
    with server.state.session_lock:
        server.state.session.add(10, 10, 10.0)
        server.state.session.add(20, 20, 9.0)

    state_now = client.get("/api/state").json()
    assert state_now.get("shooting_status") == "finished"
