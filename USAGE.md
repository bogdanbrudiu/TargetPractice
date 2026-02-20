## Usage Cheatsheet

### Start server (USB camera) — config-driven
```bash
cd TargetPractice
python3 -m pip install -r requirements.txt
python3 -m targetweb.server --camera-index 0 --latest-dir ./run --host 127.0.0.1 --port 8000 --no-open
```

### Config
If you have an `ha.ini` file compatible with HomeLESS-style options, place or copy it to `config/ha.ini` and run:
```bash
python3 -m targetweb.server --config ./config/ha.ini
```

Start from the included example:

```bash
cp config/ha.ini.example config/ha.ini
```

### USB camera (respecting VideoWidth/Height from ha.ini if set)
```bash
python3 -m targetweb.server --camera-index 0 --latest-dir ./run --host 127.0.0.1 --port 8000 --no-open
```

### Chromecast
```bash
python3 -m targetweb.server --device "Your Chromecast"
```

### Run tests
```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

### Attribution reminder
If you publish this repo, consider keeping a short acknowledgements section for inspiration sources (e.g. HomeLESS, LaserGunTargetCaster) and verify any upstream license requirements if you reuse code/assets (or redistribute bundled third-party data files under `data/`).