from __future__ import annotations

import argparse
import csv
from pathlib import Path


def sync_hit_labels(debug_dir: Path) -> dict:
    csv_path = debug_dir / "hits.csv"
    good_dir = debug_dir / "good"
    false_dir = debug_dir / "false"

    good_dir.mkdir(parents=True, exist_ok=True)
    false_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return {"ok": False, "reason": "missing hits.csv", "updated": 0, "rows": 0}

    good_names = {p.name for p in good_dir.glob("*.jpg") if p.is_file()}
    false_names = {p.name for p in false_dir.glob("*.jpg") if p.is_file()}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "image_file" not in fieldnames:
        return {"ok": False, "reason": "hits.csv missing image_file column", "updated": 0, "rows": len(rows)}
    if "label" not in fieldnames:
        fieldnames.append("label")
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    updated = 0
    for row in rows:
        img_name = (row.get("image_file") or "").strip()
        if not img_name:
            continue

        new_label = None
        if img_name in good_names:
            new_label = "good"
        elif img_name in false_names:
            new_label = "false"

        if new_label is None:
            continue
        if (row.get("label") or "").strip().lower() == new_label:
            continue

        row["label"] = new_label
        updated += 1

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "ok": True,
        "updated": updated,
        "rows": len(rows),
        "good_files": len(good_names),
        "false_files": len(false_names),
        "csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync hit labels in hits.csv based on files moved to good/false folders.")
    parser.add_argument(
        "--debug-dir",
        default="run/hit_debug",
        help="Path to hit debug directory containing hits.csv and good/false folders.",
    )
    args = parser.parse_args()

    result = sync_hit_labels(Path(args.debug_dir).resolve())
    if not result.get("ok", False):
        print(f"sync failed: {result.get('reason', 'unknown error')}")
        raise SystemExit(1)

    print(
        "synced labels: "
        f"updated={result.get('updated', 0)} "
        f"rows={result.get('rows', 0)} "
        f"good_files={result.get('good_files', 0)} "
        f"false_files={result.get('false_files', 0)}"
    )


if __name__ == "__main__":
    main()
