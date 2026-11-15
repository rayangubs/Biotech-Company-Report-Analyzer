"""Edit-distance metrics used by the benchmark (no third-party dependency)."""
from __future__ import annotations

import re
from typing import Sequence


def normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace; the usual pre-step before CER/WER."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, normalize: bool = True) -> float:
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def wer(ref: str, hyp: str, normalize: bool = True) -> float:
    if normalize:
        ref, hyp = normalize_text(ref), normalize_text(hyp)
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)
