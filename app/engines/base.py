"""Common interface for every OCR engine, so the pipeline never depends on one vendor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OCRLine:
    text: str
    conf: float                       # 0..1 (engine-specific meaning, see each engine)
    box: tuple[int, int, int, int]    # x0, y0, x1, y1 in the coordinates of the image given to the engine

    def to_dict(self) -> dict:
        return {"text": self.text, "confidence": round(float(self.conf), 4), "box": [int(v) for v in self.box]}


class OCREngine(ABC):
    name = "base"

    @abstractmethod
    def read_region(self, img_bgr: np.ndarray, multiline: bool = True) -> list[OCRLine]:
        """Recognise text in a crop. `multiline=False` means the crop holds exactly one text line."""

    def read_text(self, img_bgr: np.ndarray, multiline: bool = True) -> str:
        return "\n".join(l.text for l in self.read_region(img_bgr, multiline))


def poly_to_box(poly) -> tuple[int, int, int, int]:
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def group_into_lines(items: list[tuple[tuple[int, int, int, int], str, float]]) -> list[OCRLine]:
    """Merge word/segment detections that sit on the same row into text lines, top to bottom."""
    items = [it for it in items if it[1].strip()]
    items.sort(key=lambda it: (it[0][1] + it[0][3]) / 2)
    rows: list[list] = []
    for box, text, conf in items:
        cy, h = (box[1] + box[3]) / 2, max(1, box[3] - box[1])
        if rows:
            r_cy = np.mean([(b[1] + b[3]) / 2 for b, _, _ in rows[-1]])
            r_h = np.mean([max(1, b[3] - b[1]) for b, _, _ in rows[-1]])
            if abs(cy - r_cy) < 0.6 * max(h, r_h):
                rows[-1].append((box, text, conf))
                continue
        rows.append([(box, text, conf)])
    lines = []
    for row in rows:
        row.sort(key=lambda it: it[0][0])
        boxes = [b for b, _, _ in row]
        lines.append(OCRLine(
            text=" ".join(t.strip() for _, t, _ in row),
            conf=float(np.mean([c for _, _, c in row])),
            box=(min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)),
        ))
    return lines
