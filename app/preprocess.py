"""Loading, geometry and image-cleaning helpers (OpenCV / NumPy only)."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import MAX_PAGE_SIDE, PDF_DPI


# ----------------------------------------------------------------------------- loading
def load_pages(data: bytes, filename: str = "") -> list[np.ndarray]:
    """Return a list of BGR pages from an image or a PDF."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        pages = _pdf_pages(data)
    else:
        pages = [_decode_image(data)]
    return [limit_size(p) for p in pages]


def _decode_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")   # phone photos carry rotation in EXIF
    except Exception as exc:                                 # noqa: BLE001
        raise ValueError(f"Cannot decode image: {exc}") from exc
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _pdf_pages(data: bytes) -> list[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PDF support needs PyMuPDF: pip install pymupdf") from exc
    pages = []
    zoom = PDF_DPI / 72.0
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            code = cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR
            pages.append(cv2.cvtColor(arr, code))
    return pages


def limit_size(img: np.ndarray, max_side: int = MAX_PAGE_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


# ----------------------------------------------------------------------------- geometry
def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def detect_page_quad(img: np.ndarray) -> np.ndarray | None:
    """Find the paper outline in a photo. Returns 4 ordered points in original coordinates or None."""
    h, w = img.shape[:2]
    s = min(1.0, 800 / max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.30 * small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return order_points(approx.reshape(4, 2) / s)
    return None


def warp_to_rect(img: np.ndarray, quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(order_points(quad), dst)
    return cv2.warpPerspective(img, m, (out_w, out_h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


# ----------------------------------------------------------------------------- cleaning
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the paper background: removes shadows and uneven phone lighting."""
    gray = to_gray(gray)
    k = max(15, (min(gray.shape[:2]) // 30) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel), (0, 0), k / 2)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def ink_mask(norm_gray: np.ndarray, level: int = 170) -> np.ndarray:
    """Boolean mask of ink on an illumination-normalised page (paper is ~255)."""
    return norm_gray < level


def remove_ruled_lines(img: np.ndarray, min_len_frac: float = 0.25, v_frac: float = 0.85) -> np.ndarray:
    """Erase long horizontal rulings and full-height vertical borders, keep handwriting.

    min_len_frac: a horizontal run longer than this fraction of the crop width is a ruling.
    v_frac:       a vertical run taller than this fraction of the crop height is a border.
    Returns gray, dark ink on white.
    """
    gray = to_gray(img)
    h, w = gray.shape
    block = max(15, (min(gray.shape[:2]) // 2) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * min_len_frac)), 1)))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, int(h * v_frac)))))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    text = cv2.subtract(bw, cv2.bitwise_or(horiz, vert))
    text = cv2.morphologyEx(text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))  # re-join strokes
    text = _drop_flat_fragments(text)
    return 255 - text


def _drop_flat_fragments(bw: np.ndarray) -> np.ndarray:
    """Delete thin, wide leftovers of slightly tilted rulings (a handwritten letter is never that flat)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    max_h = max(4, int(0.03 * bw.shape[0]))
    for i in range(1, n):
        w, h = stats[i][cv2.CC_STAT_WIDTH], stats[i][cv2.CC_STAT_HEIGHT]
        if h <= max_h and w >= 25 and w >= 5 * h:
            out[labels == i] = 0
    return out


def _split_tall_runs(runs: list[tuple[int, int]], prof: np.ndarray, min_height: int) -> list[tuple[int, int]]:
    """Split runs that are much taller than the shortest line (touching descenders/ascenders) at profile valleys."""
    heights = [e - s for s, e in runs if e - s >= min_height]
    if len(heights) < 2:
        return runs
    ref = min(heights)
    out: list[tuple[int, int]] = []

    def split(s: int, e: int, depth: int = 0) -> None:
        if e - s <= 1.6 * ref or depth > 3:
            out.append((s, e))
            return
        seg = np.convolve(prof[s:e], np.ones(5) / 5, mode="same")
        lo, hi = int(0.4 * ref), len(seg) - int(0.4 * ref)
        if hi <= lo:
            out.append((s, e))
            return
        cut = lo + int(np.argmin(seg[lo:hi]))
        if seg[cut] > 0.6 * float(np.mean(seg)):          # no real valley: leave it alone
            out.append((s, e))
            return
        split(s, s + cut, depth + 1)
        split(s + cut, e, depth + 1)

    for s, e in runs:
        split(s, e)
    return out


def segment_lines(img: np.ndarray, min_height: int = 10, merge_gap_ratio: float = 0.12,
                  pad_x: int = 6, pad_y: int = 4) -> list[tuple[int, int, int, int]]:
    """Split a multi-line crop into text-line boxes (x0, y0, x1, y1) using a horizontal projection profile.

    Used by TrOCREngine, which requires one crop per text line.
    Other engines (Paddle, Tesseract) handle multi-line input natively and do not need this.
    """
    gray = to_gray(img)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    prof = (bw > 0).sum(axis=1).astype(float)
    if prof.max() <= 0:
        return []
    k = 3
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    mask = prof > max(1.5, 0.06 * prof.max())

    runs, start = [], None
    for y, v in enumerate(mask):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append((start, y))
            start = None
    if start is not None:
        runs.append((start, h))
    if not runs:
        return []

    med = float(np.median([e - s for s, e in runs]))
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap_ratio * med:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    merged = _split_tall_runs(merged, prof, min_height)

    boxes = []
    for s, e in merged:
        if e - s < min_height:
            continue
        cols = np.where((bw[s:e] > 0).sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        boxes.append((max(0, int(cols[0]) - pad_x), max(0, s - pad_y),
                      min(w, int(cols[-1]) + pad_x + 1), min(h, e + pad_y)))
    return boxes


def clear_border(gray: np.ndarray, px: int = 3) -> np.ndarray:
    """Whiten a thin frame so form borders that fall inside a crop are not read as text."""
    out = gray.copy()
    px = min(px, out.shape[0] // 3, out.shape[1] // 3)
    if px > 0:
        out[:px, :], out[-px:, :], out[:, :px], out[:, -px:] = 255, 255, 255, 255
    return out


def is_dash(gray: np.ndarray, dark: int = 140) -> bool:
    """True when the only ink is one short, flat stroke (the '-' people write for 'none')."""
    bw = (gray < dark).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = [stats[i] for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 6]
    if len(comps) != 1:
        return False
    w, h = comps[0][cv2.CC_STAT_WIDTH], comps[0][cv2.CC_STAT_HEIGHT]
    return h <= 0.25 * gray.shape[0] and w >= 2.5 * h
