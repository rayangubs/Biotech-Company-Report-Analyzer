"""PaddleOCR (PP-OCR detector + recognizer). Works with the 3.x `predict` API and the 2.x `ocr` API.

Install:  pip install paddlepaddle paddleocr      (GPU build of paddlepaddle if you have CUDA)
Confidence = the recogniser's per-line score.
"""
from __future__ import annotations

import numpy as np

from ..config import PADDLE_LANG
from .base import OCREngine, OCRLine, group_into_lines, poly_to_box


def _major_version() -> int:
    try:
        import paddleocr
        return int(str(getattr(paddleocr, "__version__", "3")).split(".")[0])
    except Exception:
        return 3


class PaddleEngine(OCREngine):
    name = "paddle"

    def __init__(self, lang: str = PADDLE_LANG):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is not installed: pip install paddlepaddle paddleocr") from exc

        if _major_version() >= 3:                       # PaddleOCR 3.x
            # enable_mkldnn=False is required on Windows to avoid a PIR/oneDNN crash.
            # Try progressively shorter arg lists so we survive future API changes.
            _inited = False
            for _kw in [
                dict(lang=lang, use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False,
                     enable_mkldnn=False),
                dict(lang=lang, enable_mkldnn=False),   # older 3.x arg subset
            ]:
                try:
                    self.ocr = PaddleOCR(**_kw)
                    _inited = True
                    break
                except (TypeError, ValueError) as _e:
                    print(f"[paddle] 3.x init kwargs {list(_kw)} failed ({_e}); trying next")
            if not _inited:
                # Absolute last resort — may re-enable oneDNN; log a warning
                print("[paddle] WARNING: falling back to PaddleOCR(lang=lang) — "
                      "oneDNN crash possible on Windows. Consider pinning paddleocr<3.")
                self.ocr = PaddleOCR(lang=lang)
            self._v3 = True
        else:                                           # PaddleOCR 2.x
            self.ocr = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)
            self._v3 = False

    def read_region(self, img_bgr: np.ndarray, multiline: bool = True) -> list[OCRLine]:
        items = self._detect_and_recognise(img_bgr)
        lines = group_into_lines(items)
        if not multiline and len(lines) > 1:           # caller promised one line: merge everything
            lines = [OCRLine(" ".join(l.text for l in lines), float(np.mean([l.conf for l in lines])),
                             (min(l.box[0] for l in lines), min(l.box[1] for l in lines),
                              max(l.box[2] for l in lines), max(l.box[3] for l in lines)))]
        return lines

    def _detect_and_recognise(self, img: np.ndarray):
        items = []
        if self._v3:
            for res in self.ocr.predict(img):
                d = res if isinstance(res, dict) else getattr(res, "json", {}).get("res", {})
                texts = d.get("rec_texts", []) or []
                scores = d.get("rec_scores", []) or []
                polys = d.get("rec_polys", None)
                if polys is None:
                    polys = d.get("dt_polys", [])
                for poly, text, score in zip(polys, texts, scores):
                    items.append((poly_to_box(poly), str(text), float(score)))
        else:
            res = self.ocr.ocr(img, cls=False)
            for page in res or []:
                for poly, (text, score) in page or []:
                    items.append((poly_to_box(poly), str(text), float(score)))
        return items