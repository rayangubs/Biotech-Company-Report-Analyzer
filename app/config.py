"""Central configuration. Every value can be overridden with an FSR_* environment variable."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(os.getenv("FSR_TEMPLATE_DIR", BASE_DIR / "templates"))
CACHE_DIR = Path(os.getenv("FSR_CACHE_DIR", BASE_DIR / "cache"))

DEFAULT_ENGINE = os.getenv("FSR_DEFAULT_ENGINE", "paddle")   # paddle | trocr | tesseract
DEFAULT_TEMPLATE = os.getenv("FSR_TEMPLATE", "fm_ser_03_rev04")

CANVAS_SCALE = float(os.getenv("FSR_SCALE", "2.5"))           # aligned page = reference size * scale
CHECKBOX_THRESHOLD = float(os.getenv("FSR_CHECK_THRESHOLD", "0.015"))
BLANK_INK_FRACTION = float(os.getenv("FSR_BLANK_INK", "0.004"))  # below this a zone is treated as empty

TROCR_MODEL = os.getenv("FSR_TROCR_MODEL", "microsoft/trocr-base-handwritten")
PADDLE_LANG = os.getenv("FSR_PADDLE_LANG", "en")
TESS_LANG = os.getenv("FSR_TESS_LANG", "eng")

PDF_DPI = int(os.getenv("FSR_PDF_DPI", "200"))
MAX_UPLOAD_MB = int(os.getenv("FSR_MAX_UPLOAD_MB", "25"))
MAX_PAGE_SIDE = int(os.getenv("FSR_MAX_PAGE_SIDE", "5000"))   # huge phone photos are downscaled first

# CORS origins — comma-separated list.  Default "*" is permissive; set FSR_CORS_ORIGINS
# to a specific origin (e.g. "https://app.example.com") in production.
CORS_ORIGINS: list[str] = os.getenv("FSR_CORS_ORIGINS", "*").split(",")

# Values that mean "nothing was written here".
PLACEHOLDERS = {"", "-", "--", "—", "–", "_", "na", "n/a", "n.a.", "none", "nil"}
