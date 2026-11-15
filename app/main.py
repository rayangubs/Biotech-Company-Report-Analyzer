"""FastAPI service.   Run:  uvicorn app.main:app --reload --port 8000    (docs at /docs)"""
from __future__ import annotations

# Paddle must be imported BEFORE torch/transformers: on Windows both ship their own cuDNN DLLs,
# and whichever loads second fails with "WinError 127" (seen as a 500 on the first /ocr call).
try:
    import paddle  # noqa: F401
except Exception as _exc:  # CPU-only installs or a broken GPU stack still fall through to the engine's own error
    print(f"[startup] paddle import failed: {_exc}")

import re
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import pipeline
from .chat import router as chat_router
from .config import CHECKBOX_THRESHOLD, CORS_ORIGINS, DEFAULT_ENGINE, DEFAULT_TEMPLATE, MAX_UPLOAD_MB, TEMPLATE_DIR
from .engines import AVAILABLE, get_engine
from .extract import router as extract_router
from .records import router as records_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the default OCR engine once at startup so the first upload isn't slow or hit by import errors."""
    try:
        get_engine(DEFAULT_ENGINE)
        print(f"[startup] engine '{DEFAULT_ENGINE}' loaded")
    except Exception as exc:
        print(f"[startup] could not load engine '{DEFAULT_ENGINE}': {exc}")
    yield


app = FastAPI(title="SBSI Field Service Report OCR API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(records_router)
app.include_router(extract_router)
app.include_router(chat_router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"   # <project root>/static/index.html


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    return data


async def _run(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:                         # missing engine / dependency
        raise HTTPException(503, str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "default_engine": DEFAULT_ENGINE, "template": DEFAULT_TEMPLATE,
            "checkbox_threshold": CHECKBOX_THRESHOLD, "templates": sorted(p.stem for p in TEMPLATE_DIR.glob("*.json"))}


@app.get("/engines")
def engines():
    return {"available": list(AVAILABLE), "default": DEFAULT_ENGINE}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), engine: str = Query(DEFAULT_ENGINE, description="paddle | trocr | tesseract"),
              use_template: bool = Query(True, description="Align to the form template and read zone by zone"),
              remove_lines: bool = Query(True, description="Erase ruled lines in multi-line zones before OCR"),
              template: str = Query(DEFAULT_TEMPLATE), use_cache: bool = Query(True)):
    data = await _read_upload(file)
    return await _run(pipeline.process_file, data, file.filename or "upload", engine=engine,
                      use_template=use_template, remove_lines=remove_lines, template_id=template, use_cache=use_cache)


@app.post("/checkboxes")
async def checkboxes(file: UploadFile = File(...), template: str = Query(DEFAULT_TEMPLATE)):
    data = await _read_upload(file)
    return await _run(pipeline.detect_checkboxes_only, data, file.filename or "upload", template_id=template)


@app.post("/debug/overlay", response_class=Response)
async def overlay(file: UploadFile = File(...), template: str = Query(DEFAULT_TEMPLATE), page: int = Query(0, ge=0)):
    """PNG showing where zones and checkboxes land on your page. Use it to tune the template JSON."""
    data = await _read_upload(file)
    png = await _run(pipeline.render_overlay, data, file.filename or "upload", template_id=template, page_index=page)
    return Response(content=png, media_type="image/png")


@app.post("/upload-scan")
async def upload_scan(file: UploadFile = File(...)):
    """Save the original scan image to static/scans/ and return a permanent URL.

    Called by the frontend right after /ocr so the scan preview survives page refreshes.
    Filenames are sanitised and deduplicated with a counter suffix if needed.
    """
    data = await _read_upload(file)
    raw_name = file.filename or "scan"

    # Sanitise: keep alphanumeric, dash, underscore, dot; replace everything else with _
    safe = unicodedata.normalize("NFKD", raw_name).encode("ascii", "ignore").decode()
    safe = re.sub(r"[^\w.\-]", "_", safe).strip("._") or "scan"

    scans_dir = STATIC_DIR / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    dest = scans_dir / safe
    # Deduplicate: scan.png → scan_1.png → scan_2.png …
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        n = 1
        while True:
            dest = scans_dir / f"{stem}_{n}{suffix}"
            if not dest.exists():
                break
            n += 1

    dest.write_bytes(data)
    return {"url": f"/scans/{dest.name}", "filename": dest.name}


# Web app. Keep this LAST: a mount at "/" would otherwise shadow the routes defined after it.
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")