"""Tesseract baseline. Weak on handwriting; included as the accuracy floor for the benchmark.

Install:  sudo apt install tesseract-ocr   (or the Windows installer) and  pip install pytesseract
"""
from __future__ import annotations

import numpy as np

from ..config import TESS_LANG
from .base import OCREngine, OCRLine, group_into_lines


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = TESS_LANG):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except ImportError as exc:
            raise RuntimeError("pip install pytesseract") from exc
        except Exception as exc:                                   # noqa: BLE001
            raise RuntimeError("Tesseract binary not found; install tesseract-ocr") from exc
        self.pt, self.lang = pytesseract, lang

    def read_region(self, img_bgr: np.ndarray, multiline: bool = True) -> list[OCRLine]:
        psm = 6 if multiline else 7
        d = self.pt.image_to_data(img_bgr, lang=self.lang, config=f"--psm {psm}", output_type=self.pt.Output.DICT)
        items = []
        for i, text in enumerate(d["text"]):
            conf = float(d["conf"][i])
            if not text.strip() or conf < 0:
                continue
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            items.append(((x, y, x + w, y + h), text, conf / 100.0))
        lines = group_into_lines(items)
        if not multiline and len(lines) > 1:
            lines = [OCRLine(
                " ".join(l.text for l in lines),
                float(np.mean([l.conf for l in lines])),
                (min(l.box[0] for l in lines), min(l.box[1] for l in lines),
                 max(l.box[2] for l in lines), max(l.box[3] for l in lines)),
            )]
        return lines
