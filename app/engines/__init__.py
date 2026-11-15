"""Engine registry. Engines are created lazily (model loading is slow) and reused."""
from __future__ import annotations

import threading

from .base import OCREngine, OCRLine

_ENGINES: dict[str, OCREngine] = {}
_ENGINE_LOCK = threading.Lock()
AVAILABLE = ("paddle", "trocr", "tesseract")


def get_engine(name: str) -> OCREngine:
    name = name.lower()
    if name not in AVAILABLE:
        raise ValueError(f"Unknown engine '{name}'. Choose from: {', '.join(AVAILABLE)}")
    if name not in _ENGINES:
        with _ENGINE_LOCK:
            # Double-checked locking: re-test inside the lock in case another thread
            # already initialised this engine while we were waiting.
            if name not in _ENGINES:
                if name == "paddle":
                    from .paddle_engine import PaddleEngine as cls
                elif name == "trocr":
                    from .trocr_engine import TrOCREngine as cls
                else:
                    from .tesseract_engine import TesseractEngine as cls
                _ENGINES[name] = cls()
    return _ENGINES[name]


__all__ = ["OCREngine", "OCRLine", "get_engine", "AVAILABLE"]
