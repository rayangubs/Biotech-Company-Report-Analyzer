"""TrOCR handwriting recogniser on single text-line crops (Hugging Face Transformers).

Install:  pip install torch transformers
TrOCR reads one line at a time, so multi-line zones are first split with a projection profile.
Confidence = exp(length-normalised beam log-probability), a rough but useful ranking signal.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..config import TROCR_MODEL
from ..preprocess import segment_lines, to_gray
from .base import OCREngine, OCRLine


class TrOCREngine(OCREngine):
    name = "trocr"

    def __init__(self, model_name: str = TROCR_MODEL, batch_size: int = 8, num_beams: int = 4):
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError("TrOCR needs: pip install torch transformers") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size, self.num_beams = batch_size, num_beams
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

    # ------------------------------------------------------------------ public
    def read_region(self, img_bgr: np.ndarray, multiline: bool = True) -> list[OCRLine]:
        gray = to_gray(img_bgr)
        h, w = gray.shape
        boxes = segment_lines(gray) if multiline else [(0, 0, w, h)]
        if not boxes:
            return []
        crops = [self._to_pil(gray[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes]
        results = self._recognise(crops)
        return [OCRLine(text=t.strip(), conf=c, box=b) for b, (t, c) in zip(boxes, results) if t.strip()]

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _to_pil(gray_crop: np.ndarray) -> Image.Image:
        crop = cv2.copyMakeBorder(gray_crop, 6, 6, 8, 8, cv2.BORDER_CONSTANT, value=255)
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))

    def _recognise(self, crops: list[Image.Image]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for i in range(0, len(crops), self.batch_size):
            batch = crops[i:i + self.batch_size]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(pixel_values, num_beams=self.num_beams, max_new_tokens=64,
                                          return_dict_in_generate=True, output_scores=True)
            texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
            seq_scores = getattr(gen, "sequences_scores", None)
            confs = [min(1.0, float(np.exp(s))) for s in seq_scores.tolist()] if seq_scores is not None else [0.0] * len(texts)
            out.extend(zip(texts, confs))
        return out
