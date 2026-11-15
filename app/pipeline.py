"""End-to-end page processing: load -> align to template -> zone OCR -> checkboxes -> JSON."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import cache
from .checkbox import detect_checkboxes, selected_by_group
from .config import (BLANK_INK_FRACTION, CANVAS_SCALE, CHECKBOX_THRESHOLD, DEFAULT_ENGINE,
                     DEFAULT_TEMPLATE)
from .engines import OCREngine, get_engine
from .nlp.normalize import is_empty as is_placeholder
from .preprocess import (clear_border, is_dash, load_pages, normalize_illumination, remove_ruled_lines,
                         to_bgr, to_gray)
from .template import Template, align_page, load_template

_LOCK = threading.Lock()          # PaddleOCR / torch models are not guaranteed thread-safe
ZONE_PAD = 2                      # reference pixels added around every zone crop
PIPELINE_VERSION = "1"            # bump to invalidate cached results after code changes


# ----------------------------------------------------------------------------- public API
def process_file(data: bytes, filename: str = "upload", engine: str = DEFAULT_ENGINE,
                 use_template: bool = True, remove_lines: bool = True, template_id: str = DEFAULT_TEMPLATE,
                 use_cache: bool = True, scale: float = CANVAS_SCALE) -> dict:
    opts = dict(engine=engine, use_template=use_template, remove_lines=remove_lines, template=template_id,
                scale=scale, thr=CHECKBOX_THRESHOLD, v=PIPELINE_VERSION)
    key = cache.make_key(data, **opts)
    if use_cache and (hit := cache.load(key)) is not None:
        hit["cached"] = True
        return hit

    t0 = time.time()
    pages = load_pages(data, filename)
    tpl = load_template(template_id) if use_template else None
    with _LOCK:
        eng = get_engine(engine)
        page_results = [process_page(p, tpl, eng, remove_lines, scale) | {"page": i + 1} for i, p in enumerate(pages)]

    result = {"filename": filename, "engine": engine, "template": template_id if use_template else None,
              "options": {"remove_lines": remove_lines, "scale": scale, "checkbox_threshold": CHECKBOX_THRESHOLD},
              "pages": page_results, "elapsed_s": round(time.time() - t0, 2), "cached": False}
    if use_cache:
        cache.save(key, result)
    return result


def detect_checkboxes_only(data: bytes, filename: str = "upload", template_id: str = DEFAULT_TEMPLATE,
                           scale: float = CANVAS_SCALE) -> dict:
    tpl = load_template(template_id)
    out = []
    for i, page in enumerate(load_pages(data, filename)):
        canvas, info = align_page(page, tpl, scale)
        norm, sx, sy = _normalise(canvas, tpl)
        cbs = detect_checkboxes(norm, tpl, sx, sy)
        out.append({"page": i + 1, "alignment": info, "checkboxes": cbs, "selected": selected_by_group(cbs)})
    return {"filename": filename, "template": template_id, "pages": out}


def render_overlay(data: bytes, filename: str = "upload", template_id: str = DEFAULT_TEMPLATE,
                   scale: float = CANVAS_SCALE, page_index: int = 0) -> bytes:
    """PNG of the aligned page with zones (blue), table cells (orange) and checkboxes (green=ticked, red=empty)."""
    tpl = load_template(template_id)
    page = load_pages(data, filename)[page_index]
    canvas, info = align_page(page, tpl, scale)
    norm, sx, sy = _normalise(canvas, tpl)
    cbs = detect_checkboxes(norm, tpl, sx, sy)
    img = canvas.copy()
    for name, z in tpl.zones.items():
        x0, y0, x1, y1 = _scaled(z["box"], sx, sy)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 80, 0), 2)
        cv2.putText(img, name, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1)
    for cell in _table_cells(tpl):
        x0, y0, x1, y1 = _scaled(cell["box"], sx, sy)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 140, 255), 1)
    side = 8.5
    for cid, c in cbs.items():
        cx, cy = c["center"]
        x0, y0, x1, y1 = _scaled([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], sx, sy)
        color = (0, 180, 0) if c["checked"] else (0, 0, 220)
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), color, 2)
    cv2.putText(img, f"alignment: {info['method']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ----------------------------------------------------------------------------- page level
def process_page(page: np.ndarray, tpl: Template | None, engine: OCREngine,
                 remove_lines: bool = True, scale: float = CANVAS_SCALE) -> dict:
    if tpl is None:                                     # template-free mode: plain full-page OCR
        norm = normalize_illumination(to_gray(page))
        lines = engine.read_region(to_bgr(norm), multiline=True)
        return {"alignment": {"method": "none", "ok": True}, "lines": [l.to_dict() for l in lines],
                "text": "\n".join(l.text for l in lines)}

    canvas, info = align_page(page, tpl, scale)
    norm, sx, sy = _normalise(canvas, tpl)

    zones = {name: _read_zone(norm, spec["box"], spec["multiline"], sx, sy, engine, remove_lines)
             for name, spec in tpl.zones.items()}
    table = _read_table(norm, tpl, sx, sy, engine)
    checkboxes = detect_checkboxes(norm, tpl, sx, sy)

    res = {"alignment": info, "zones": zones, "table": table,
           "checkboxes": checkboxes, "selected": selected_by_group(checkboxes)}
    low = [k for k, v in checkboxes.items() if v["box_match"] < 0.3]
    if low:
        res["warnings"] = [f"printed box not located for: {', '.join(low)} (check alignment/zones)"]
    return res


# ----------------------------------------------------------------------------- helpers
def _normalise(canvas: np.ndarray, tpl: Template):
    norm = normalize_illumination(to_gray(canvas))
    return norm, canvas.shape[1] / tpl.ref_w, canvas.shape[0] / tpl.ref_h


def _scaled(box, sx, sy, pad: float = 0.0, shape=None):
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = int((x0 - pad) * sx), int((y0 - pad) * sy), int(round((x1 + pad) * sx)), int(round((y1 + pad) * sy))
    if shape is not None:
        H, W = shape[:2]
        X0, Y0, X1, Y1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    return X0, Y0, X1, Y1


def _read_zone(norm: np.ndarray, box, multiline: bool, sx: float, sy: float,
               engine: OCREngine, remove_lines: bool, pad: float = ZONE_PAD) -> dict:
    x0, y0, x1, y1 = _scaled(box, sx, sy, pad, norm.shape)
    crop = norm[y0:y1, x0:x1]
    if crop.size == 0:
        return {"text": "", "confidence": 0.0, "lines": [], "empty": True}
    if remove_lines:                                    # erase form rulings, then the frame that clips borders
        crop = remove_ruled_lines(crop, min_len_frac=0.25 if multiline else 0.6, v_frac=0.5 if multiline else 0.9)
    crop = clear_border(crop, 3)

    ink_frac = float((crop < 140).mean())
    if ink_frac < BLANK_INK_FRACTION:                   # nothing written: skip the slow OCR call
        return {"text": "", "confidence": 1.0, "lines": [], "empty": True, "ink": round(ink_frac, 4)}
    if not multiline and is_dash(crop):                 # a lone dash means "none"
        return {"text": "-", "confidence": 1.0, "lines": [], "empty": True, "ink": round(ink_frac, 4)}

    lines = engine.read_region(to_bgr(crop), multiline=multiline)
    text = "\n".join(l.text for l in lines)
    conf = float(np.mean([l.conf for l in lines])) if lines else 0.0
    return {"text": text, "confidence": round(conf, 4), "lines": [l.to_dict() for l in lines],
            "empty": is_placeholder(text) or not text.strip(), "ink": round(ink_frac, 4)}


def _table_cells(tpl: Template) -> list[dict]:
    t = tpl.table
    if not t:
        return []
    row_h = (t["y1"] - t["y0"]) / t["rows"]
    cells = []
    for r in range(t["rows"]):
        for col, (cx0, cx1) in t["columns"].items():
            cells.append({"row": r, "col": col, "box": [cx0, t["y0"] + r * row_h + 1, cx1, t["y0"] + (r + 1) * row_h - 1]})
    return cells


def _read_table(norm, tpl: Template, sx, sy, engine: OCREngine) -> dict | None:
    if not tpl.table:
        return None
    rows: dict[int, dict] = {}
    for cell in _table_cells(tpl):
        z = _read_zone(norm, cell["box"], False, sx, sy, engine, remove_lines=True, pad=-1)
        rows.setdefault(cell["row"], {})[cell["col"]] = {"text": z["text"], "confidence": z["confidence"], "empty": z["empty"]}
    kept = [r for r in rows.values() if not all(c["empty"] for c in r.values())]
    return {"name": tpl.table["name"], "rows": kept, "all_empty": not kept}
