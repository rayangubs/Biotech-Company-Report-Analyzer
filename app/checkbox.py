"""Checkbox state detection on an aligned, illumination-normalised page.

Two steps per box:
  1. refine the expected position by matching a square-outline template (tolerates small alignment drift);
  2. measure the ink fraction inside the box interior (the printed border is excluded).
A tick or cross inside (even one that spills over the edge) pushes the interior ink fraction up.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import CHECKBOX_THRESHOLD
from .preprocess import ink_mask
from .template import Template

DEFAULT_BOX_SIZE = 8.5       # printed box side, reference pixels
INNER_MARGIN = 0.24          # fraction of the box ignored on each side (skips the printed border)
SEARCH_RADIUS = 5            # reference pixels searched when refining the box position
MIN_MATCH = 0.30             # minimum outline-match score to accept a refined position


def refine_center(norm_gray: np.ndarray, cx: float, cy: float, size: float, sx: float, sy: float) -> tuple[float, float, float]:
    """Return (cx, cy, match_score) in canvas pixels after snapping to the printed square."""
    X, Y = cx * sx, cy * sy
    side = max(6, int(round(size * sx)))
    thick = max(2, int(round(side * 0.14)))
    ring = np.zeros((side + 2, side + 2), np.float32)
    cv2.rectangle(ring, (1, 1), (side, side), 1.0, thick)

    r = int(SEARCH_RADIUS * sx)
    pad = side // 2 + 2
    x0, y0 = int(X - r - pad), int(Y - r - pad)
    x1, y1 = int(X + r + pad), int(Y + r + pad)
    H, W = norm_gray.shape
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return X, Y, 0.0
    window = (255 - norm_gray[y0:y1, x0:x1]).astype(np.float32) / 255.0
    if window.shape[0] < ring.shape[0] or window.shape[1] < ring.shape[1]:
        return X, Y, 0.0
    res = cv2.matchTemplate(window, ring, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < MIN_MATCH:
        return X, Y, float(mx)
    return x0 + loc[0] + ring.shape[1] / 2.0, y0 + loc[1] + ring.shape[0] / 2.0, float(mx)


def interior_ink(ink: np.ndarray, X: float, Y: float, size: float, sx: float, sy: float) -> float:
    half_w = size * sx * (0.5 - INNER_MARGIN)
    half_h = size * sy * (0.5 - INNER_MARGIN)
    H, W = ink.shape
    x0, x1 = int(max(0, round(X - half_w))), int(min(W, round(X + half_w)))
    y0, y1 = int(max(0, round(Y - half_h))), int(min(H, round(Y + half_h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(ink[y0:y1, x0:x1].mean())


def detect_checkboxes(norm_gray: np.ndarray, tpl: Template, sx: float, sy: float,
                      threshold: float = CHECKBOX_THRESHOLD) -> dict:
    ink = ink_mask(norm_gray)
    out = {}
    for cid, spec in tpl.checkboxes.items():
        cx, cy = spec["center"]
        size = spec.get("size", DEFAULT_BOX_SIZE)
        X, Y, match = refine_center(norm_gray, cx, cy, size, sx, sy)
        score = interior_ink(ink, X, Y, size, sx, sy)
        out[cid] = {
            "label": spec["label"],
            "group": spec["group"],
            "checked": bool(score >= threshold),
            "score": round(score, 4),
            "confidence": round(min(1.0, abs(score - threshold) / max(threshold, 1e-6)), 3),
            "box_match": round(match, 3),          # low value = printed box not found, check alignment
            "center": [round(X / sx, 1), round(Y / sy, 1)],
        }
    return out


def selected_by_group(checkboxes: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for cid, c in checkboxes.items():
        groups.setdefault(c["group"], [])
        if c["checked"]:
            groups[c["group"]].append(cid)
    return groups
