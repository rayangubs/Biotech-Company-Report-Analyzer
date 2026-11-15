"""Shared JSONL record store.

A single in-memory cache is kept after the first read.  It is invalidated
whenever a new record is written, so queries always see the latest data
without hitting the filesystem on every request.

Both ``records.py`` and ``nlp/query_engine.py`` import from here so the
read/deduplicate logic lives in exactly one place.
"""
from __future__ import annotations

import json
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "data" / "records.jsonl"

# In-memory cache.  None means "not yet loaded / invalidated".
_cache: dict[str, dict] | None = None


def read_records() -> dict[str, dict]:
    """Return {id: record} dict, rebuilt from JSONL on first call or after invalidation."""
    global _cache
    if _cache is not None:
        return _cache
    out: dict[str, dict] = {}
    if STORE.exists():
        for line in STORE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["id"]] = rec          # last write wins (append-log dedup)
            except (json.JSONDecodeError, KeyError):
                pass
    _cache = out
    return _cache


def load_records() -> list[dict]:
    """Return records as a list (preserves insertion order of unique IDs)."""
    return list(read_records().values())


def append_record(rec: dict) -> None:
    """Persist *rec* to the JSONL file and invalidate the in-memory cache."""
    global _cache
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _cache = None   # force reload on next read
