"""Form template: zone/checkbox layout plus alignment of a page photo onto the template."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import CANVAS_SCALE, TEMPLATE_DIR
from .preprocess import detect_page_quad, to_gray, warp_to_rect


@dataclass
class Template:
    id: str
    ref_w: int
    ref_h: int
    zones: dict
    checkboxes: dict
    table: dict | None = None
    image: np.ndarray | None = field(default=None, repr=False)


_CACHE: dict[str, Template] = {}


def load_template(template_id: str, template_dir: Path = TEMPLATE_DIR) -> Template:
    if template_id in _CACHE:
        return _CACHE[template_id]
    json_path = Path(template_dir) / f"{template_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Template not found: {json_path}")
    spec = json.loads(json_path.read_text(encoding="utf-8"))
    image = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = Path(template_dir) / f"{template_id}{ext}"
        if p.exists():
            image = cv2.imread(str(p))
            break
    tpl = Template(id=spec["id"], ref_w=spec["reference_size"][0], ref_h=spec["reference_size"][1],
                   zones=spec["zones"], checkboxes=spec["checkboxes"], table=spec.get("table"), image=image)
    _CACHE[template_id] = tpl
    return tpl


def canvas_size(tpl: Template, scale: float = CANVAS_SCALE) -> tuple[int, int]:
    return int(round(tpl.ref_w * scale)), int(round(tpl.ref_h * scale))


def align_page(page: np.ndarray, tpl: Template, scale: float = CANVAS_SCALE,
               min_inliers: int = 25) -> tuple[np.ndarray, dict]:
    """Warp `page` onto the template frame (reference size * scale).

    1) SIFT features + RANSAC homography against the template image (handles rotation, skew, perspective).
    2) If that fails: page-outline detection + resize.  3) Last resort: plain resize.
    """
    out_w, out_h = canvas_size(tpl, scale)

    if tpl.image is not None:
        # Rotate landscape photos to portrait before attempting SIFT so the
        # feature detector sees the same orientation as the template image.
        h, w = page.shape[:2]
        if w > h:
            page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)
        res = _align_sift(page, tpl, out_w, out_h, min_inliers)
        if res is not None:
            canvas, inliers = res
            return canvas, {"method": "sift", "ok": True, "quality": "high", "inliers": inliers}
    else:
        h, w = page.shape[:2]
        if w > h:
            page = cv2.rotate(page, cv2.ROTATE_90_CLOCKWISE)

    quad = detect_page_quad(page)
    if quad is not None:
        return warp_to_rect(page, quad, out_w, out_h), {
            "method": "page_outline", "ok": True, "quality": "medium",
            "warning": "SIFT template matching failed; used page outline. Zones may be slightly misplaced."}
    return cv2.resize(page, (out_w, out_h), interpolation=cv2.INTER_AREA), {
        "method": "resize", "ok": False, "quality": "low",
        "warning": "Template matching failed; page resized blindly. Zones may be misplaced."}


def _align_sift(page: np.ndarray, tpl: Template, out_w: int, out_h: int, min_inliers: int):
    h, w = page.shape[:2]
    d = min(1.0, 1800 / max(h, w))                     # detect features on a downscaled copy
    small = cv2.resize(page, (int(w * d), int(h * d)), interpolation=cv2.INTER_AREA) if d < 1 else page

    sift = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = sift.detectAndCompute(to_gray(small), None)
    # Use a copy so accidental in-place ops never corrupt the cached template image.
    k2, d2 = sift.detectAndCompute(to_gray(tpl.image.copy()), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < min_inliers:
        return None

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or int(mask.sum()) < min_inliers:
        return None

    th, tw = tpl.image.shape[:2]
    # template pixels -> canvas pixels
    T = np.diag([out_w / tw, out_h / th, 1.0])
    # original page pixels -> downscaled page pixels
    D = np.diag([d, d, 1.0])
    canvas = cv2.warpPerspective(page, T @ H @ D, (out_w, out_h),
                                 flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
    return canvas, int(mask.sum())
