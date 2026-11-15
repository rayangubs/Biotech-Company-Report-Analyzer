# =============================================================================
# SBSI Field Service Report OCR – Dockerfile
# =============================================================================
#
# Build:   docker build -t fsr-ocr .
# Run:     docker run -p 8000:8000 fsr-ocr
# Or use:  docker compose up      (see docker-compose.yml)
#
# The image is based on a slim Python 3.11 base.
# System packages needed by OpenCV and PaddleOCR are installed first (they
# rarely change), then Python packages (change occasionally), then app code
# (changes most often).  This ordering maximises Docker's layer cache — if you
# only change a source file, Docker skips the slow install steps on rebuild.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 – base OS + system libraries
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (you see logs in real time)
ENV PYTHONUNBUFFERED=1

# Prevent pip from saving a local cache inside the image (saves space)
ENV PIP_NO_CACHE_DIR=1

# Tell OpenCV not to look for a display (we're running headless on a server)
ENV OPENCV_IO_ENABLE_OPENEXR=0

WORKDIR /app

# Install system packages:
#   libgl1 + libglib2.0-0  — OpenCV needs these even in headless mode
#   tesseract-ocr           — the Tesseract binary (the pytesseract pip package
#                             is just a Python wrapper; the binary must be installed separately)
#   libgomp1                — OpenMP, used by PaddlePaddle for CPU parallelism
#   curl                    — useful for health-check scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2 – Python dependencies
# (copied and installed before the app code so this layer is cached separately)
# -----------------------------------------------------------------------------
COPY requirements.txt .

# Install the core requirements (FastAPI, OpenCV, PyMuPDF, etc.)
# PaddleOCR and TrOCR are large optional engines.  They are installed here
# so the image works out of the box.  Remove lines you don't need to slim
# down the image size.
RUN pip install --upgrade pip \
    # Core app requirements
    && pip install -r requirements.txt \
    # --- PaddleOCR (CPU build; swap paddlepaddle for paddlepaddle-gpu for CUDA) ---
    && pip install paddlepaddle paddleocr \
    # --- TrOCR: PyTorch CPU build + HuggingFace Transformers ---
    # The CPU-only torch wheel is much smaller than the CUDA build.
    # If your server has a GPU, replace this with the CUDA wheel from pytorch.org.
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install transformers \
    # --- pytesseract Python wrapper (the binary was installed above via apt) ---
    && pip install pytesseract \
    # --- rapidfuzz for faster fuzzy matching in the NLP layer ---
    && pip install rapidfuzz

# -----------------------------------------------------------------------------
# Stage 3 – Application code
# -----------------------------------------------------------------------------
COPY app/           app/
COPY templates/     templates/
COPY static/        static/
COPY data/lexicon/  data/lexicon/
COPY .gitignore     .
# records.jsonl is intentionally NOT copied into the image — it lives in a
# Docker volume (see docker-compose.yml) so confirmed records persist across
# container restarts.

# Create the directories the app writes to at runtime.
# They will be bind-mounted to the host in production, but need to exist
# inside the image too for standalone `docker run` usage.
RUN mkdir -p cache data static/scans

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
# Run as a non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Healthcheck: Docker will mark the container unhealthy if the API stops
# responding.  The interval is long because model loading takes time on startup.
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI server.
# --host 0.0.0.0 makes it reachable from outside the container.
# Remove --reload in production (it's for development only).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
