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

### Debug false triggers
Each recorded trigger is saved to `run/hit_debug/`.

- JPEG filename includes timestamp, source, and trigger coordinates, for example `20260608_143015_123_auto_x318_y227_s95.jpg`
- `run/hit_debug/hits.csv` contains one row per trigger with detector settings and empty `label` / `notes` columns you can edit later
- A practical workflow is to review the JPEGs at the end of the day, mark `label=good` or `label=false` in the CSV, then use those examples to tune thresholds with evidence

### Easier labeling (no manual CSV edit)
You can label by moving images into folders and then syncing:

1. Move real hits into `run/hit_debug/good/`
2. Move false hits into `run/hit_debug/false/`
3. Run sync command:

```bash
python -m targetweb.hit_debug_labels --debug-dir run/hit_debug
```

This updates only the `label` column in `hits.csv` (`good` / `false`) for matching filenames.

### Attribution reminder
If you publish this repo, consider keeping a short acknowledgements section for inspiration sources (e.g. HomeLESS, LaserGunTargetCaster) and verify any upstream license requirements if you reuse code/assets (or redistribute bundled third-party data files under `data/`).