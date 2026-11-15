"""Tiny on-disk result cache so re-running a page never repeats slow OCR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import CACHE_DIR


def make_key(data: bytes, **options) -> str:
    h = hashlib.sha256(data)
    h.update(json.dumps(options, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]


def load(key: str, cache_dir: Path = CACHE_DIR) -> dict | None:
    p = Path(cache_dir) / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def save(key: str, result: dict, cache_dir: Path = CACHE_DIR) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / f"{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
