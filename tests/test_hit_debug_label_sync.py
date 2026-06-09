from pathlib import Path

from targetweb.hit_debug_labels import sync_hit_labels


def test_sync_hit_labels_from_good_false_folders(tmp_path: Path):
    dbg = tmp_path / "hit_debug"
    dbg.mkdir(parents=True, exist_ok=True)

    csv_path = dbg / "hits.csv"
    csv_path.write_text(
        "image_file,label,notes\n"
        "a.jpg,,\n"
        "b.jpg,,\n"
        "c.jpg,good,\n",
        encoding="utf-8",
    )

    good = dbg / "good"
    false = dbg / "false"
    good.mkdir(parents=True, exist_ok=True)
    false.mkdir(parents=True, exist_ok=True)

    (good / "a.jpg").write_bytes(b"x")
    (false / "b.jpg").write_bytes(b"x")

    result = sync_hit_labels(dbg)

    assert result["ok"] is True
    assert result["updated"] == 2

    content = csv_path.read_text(encoding="utf-8")
    assert "a.jpg,good," in content
    assert "b.jpg,false," in content
    assert "c.jpg,good," in content
