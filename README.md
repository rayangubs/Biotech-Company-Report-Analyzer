# SBSI Field Service Report – Local OCR Service

Local-only OCR foundation for the NLP project. No image ever leaves your machine.

```
upload (image/PDF) -> page load -> align to form template (SIFT + RANSAC)
   -> illumination normalisation -> per-zone crop -> ruling/border removal
   -> OCR engine (paddle | trocr | tesseract) -> JSON
   -> checkbox detector (ticked / empty) -> JSON
```

## Running with Docker (recommended — no Python setup required)

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / Mac / Linux), then:

```bash
git clone https://github.com/YourUsername/fsr-ocr.git
cd fsr-ocr

# First run: builds the image (takes ~5–10 min, downloads ML models)
docker compose up

# App is now live at http://localhost:8000
# API docs at      http://localhost:8000/docs
```

Subsequent starts are instant (the image is cached):
```bash
docker compose up          # start
docker compose down        # stop
docker compose up --build  # rebuild after changing requirements.txt or Dockerfile
```

Your confirmed records and uploaded scans are stored in your local `data/` and `static/scans/` folders via Docker volumes — they survive container restarts and rebuilds.

---

## Running without Docker (manual setup)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# then the engine(s) you want, see the comments in requirements.txt
uvicorn app.main:app --reload --port 8000                # docs: http://localhost:8000/docs
```

## Endpoints
| Method | Path | What it does |
|---|---|---|
| GET | `/health`, `/engines` | status, templates, available engines |
| POST | `/ocr?engine=paddle&remove_lines=true` | full result: zones, parts table, checkboxes |
| POST | `/checkboxes` | only the checkbox states (no OCR engine needed) |
| POST | `/debug/overlay` | PNG with zones + checkbox results drawn on your page |

```bash
curl -F "file=@report.jpg" "http://localhost:8000/ocr?engine=trocr" | jq .pages[0].zones.actions_taken
```
Set `use_template=false` for plain full-page OCR. Results are cached in `cache/` (delete files to force a re-run, or pass `use_cache=false`).

## First things to do (in this order)
1. **Replace `templates/fm_ser_03_rev04.png` with a clean scan of the blank form** (the shipped file is the handwritten mock; it works for alignment but a blank is better). Each form revision needs its own `<id>.json` + `<id>.png`.
2. **Calibrate the layout:** `python scripts/calibrate_zones.py your_photo.jpg -o overlay.png` and adjust `zones`, `table` and checkbox `center` values in `templates/*.json` until the boxes sit on the right places. Coordinates are in the `reference_size` pixel frame.
3. **Build the benchmark** (10–15 reports, mix of scans/photos/PDF, PM and corrective):
   ```bash
   python scripts/benchmark.py --images data/benchmark --init-gold data/gold.json   # then fill in gold.json by hand
   python scripts/benchmark.py --images data/benchmark --gold data/gold.json --engines paddle trocr tesseract --sweep
   ```
   It prints mean CER/WER per engine, writes `benchmark_results.csv` per zone, and shows checkbox accuracy for several thresholds. Set the winner with `FSR_DEFAULT_ENGINE` and the threshold with `FSR_CHECK_THRESHOLD`.
4. Try `remove_lines=false` vs `true` in the benchmark; whichever scores better per engine is your default.

## Configuration (environment variables)
`FSR_DEFAULT_ENGINE`, `FSR_TEMPLATE`, `FSR_SCALE` (canvas upscale, default 2.5), `FSR_CHECK_THRESHOLD` (default 0.015),
`FSR_BLANK_INK`, `FSR_TROCR_MODEL` (e.g. `microsoft/trocr-large-handwritten`), `FSR_PADDLE_LANG`, `FSR_PDF_DPI`, `FSR_MAX_UPLOAD_MB`, `FSR_CACHE_DIR`, `FSR_TEMPLATE_DIR`.

## Output shape (abridged)
```json
{"pages": [{"alignment": {"method": "sift", "ok": true, "inliers": 3346},
  "zones": {"institution": {"text": "...", "confidence": 0.81, "empty": false, "lines": [...]},
            "actions_taken": {"text": "Check Machine\nOpen Cover\n...", "empty": false}},
  "table": {"name": "parts_used", "rows": [], "all_empty": true},
  "checkboxes": {"six_month_pm": {"checked": true, "score": 0.12, "confidence": 0.8, "box_match": 0.56}},
  "selected": {"job_type": ["preventive_maintenance", "six_month_pm"], "status_after_service": ["operational"]}}]}
```
`empty` is true for blank zones and for placeholders (`NA`, `-`, `none`). Check `alignment.ok` and `warnings`: if SIFT matching failed the zones may be misplaced.

## Known limits
* Checkbox detection is a heuristic (interior ink after snapping to the printed square). Ticks that stay almost entirely outside the box are the weak spot; the threshold must be tuned on your own scans (use `--sweep`). A small CNN on cropped boxes is the upgrade path once you have labelled crops.
* TrOCR reads one line at a time and squeezes each crop to 384×384, so very long lines lose resolution. Compare against Paddle on your benchmark.
* A page with no readable printed content (very blurry or cropped photo) will fail template matching and fall back to page-outline detection.
