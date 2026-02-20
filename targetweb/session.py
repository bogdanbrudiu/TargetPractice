from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class Shot:
    x: float
    y: float
    score: float
    time: str


class Session:
    def __init__(self, out_dir: Path, target_filename: str, cx: float, cy: float, scale: float) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%y%m%d%H%M")
        self.csv = self.out_dir / f"{ts}.csv"
        self.shots: List[Shot] = []
        self.total: float = 0.0
        # header similar to HomeLESS
        with self.csv.open("w", encoding="utf-8") as f:
            f.write(f"CX|{cx}\n")
            f.write(f"CY|{cy}\n")
            f.write(f"Target|{target_filename}\n")
            f.write(f"Scale|{scale}\n")
            f.write("TriggerId\n")
            f.write("TriggerTime\n")

    def add(self, x: float, y: float, score: float) -> None:
        t = datetime.now().strftime("%H:%M:%S")
        shot = Shot(x=x, y=y, score=score, time=t)
        self.shots.append(shot)
        self.total += score
        with self.csv.open("a", encoding="utf-8") as f:
            f.write(f"Shot|{int(x)}|{int(y)}|{score:.1f}|{t}\n")

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "shots": [asdict(s) for s in self.shots],
        }
