"""POST /extract: normalized fields for one /ocr page result.

Body: the /ocr result of one page (only its "zones" object is used), e.g. res.pages[0]
for a PDF. Returns normalized values, validation warnings, per-line action tags and
remark kinds. Nothing is stored here.

Kept as a separate module so the NLP layer can be imported and tested independently
of the full FastAPI app.
"""
from fastapi import APIRouter

from .nlp.normalize import extract_report

router = APIRouter()


@router.post("/extract")
def extract(body: dict):
    return extract_report(body.get("zones") or {})
