# TargetWeb: Laser Target Practice Web App

Prototype FastAPI app that:
- Parses HomeLESS `.tgt` targets and uses the same ring-diameter scoring model.
- Detects laser hits via a simple bright-spot detector (inspired by LaserGunTargetCaster but implemented independently).
- Overlays target "ghost" and hit/score on frames, saves `run/frames/latest.jpg`.
- Serves a webpage showing the frame and scoreboard; optional Chromecast device param.

## Quick Start

Install deps (recommended: use a virtualenv; but you can also run without one):

```bash
cd TargetPractice
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Without a venv (system Python):

```bash
cd TargetPractice
python3 -m pip install -r requirements.txt
```

Create a local config (optional):

```bash
cp config/ha.ini.example config/ha.ini
```


Run with a USB camera:

```bash
python3 -m targetweb.server  --target data/targets/10m_air_pistol.tgt  --camera-index 0  --latest-dir run  --host 127.0.0.1 --port 8000  --no-open
```

Open http://localhost:8000/ in a browser.

Optional Chromecast (best-effort):

```bash
python -m targetweb.server --target data/targets/10m_air_pistol.tgt --device "Living Room TV"
```

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

## Acknowledgements / Inspiration

This project takes inspiration from existing open-source projects in this space:

- HomeLESS: target parsing and scoring model compatibility (`.tgt`, ring model)
- LaserGunTargetCaster: general approach to bright-spot/laser hit detection

The code here is a separate implementation (not a direct copy/paste), but you should still keep attribution and comply with any upstream license requirements if you incorporate code/assets.

In this repo, the application code is original; the only third-party content included is the target/weapon definition files under `data/` (see `ACKNOWLEDGEMENTS.md` for notes).

See `ACKNOWLEDGEMENTS.md` for a place to keep source links and notes.

## License

No license is granted for the original code in this repository at this time (all rights reserved). If you want to use, modify, or redistribute the code, please contact the repository owner for permission.

Note: this repository includes third-party target/weapon definition files under `data/` which are governed by their upstream license terms (see `ACKNOWLEDGEMENTS.md` and `THIRD_PARTY_LICENSES/`).

## Notes
- Circle target parsing: line 1 name, line 2 type (0 metric, 2 imperial), next 10 lines diameters, line 13 is marked ring index (1-based). Imperial diameters converted to mm.
- Scaling emulates HomeLESS: pixel-to-mm scale derives from frame height and the largest diameter.
- Detection is intentionally simple for prototyping and can be swapped out.
- Every recorded trigger is also stored under `run/hit_debug/` with coordinates in the JPEG filename and a `hits.csv` index for later manual labeling (`label`, `notes`).
