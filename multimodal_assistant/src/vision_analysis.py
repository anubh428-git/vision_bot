"""
vision_analysis.py
-------------------
Extracts structured, checkable evidence from an image -- this is the
"understand visual content" half of the assistant. Rather than a single
end-to-end VQA call, we deliberately decompose vision understanding into
independent signals (caption, zero-shot concept tags, OCR text, embedding)
so downstream stages (validator, ambiguity handler) have concrete evidence
to check claims against, instead of one opaque model output to trust blindly.

Two backends:
  * "hf"   -- BLIP for captioning, CLIP for zero-shot tagging + embeddings,
              pytesseract for OCR. All open-source / open-weight.
  * "mock" -- deterministic, dependency-free analyzer used for testing the
              orchestration logic (ambiguity handling, planning, validation)
              without needing to download model weights or a GPU.
"""
import sys, os
import hashlib
from functools import lru_cache
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.schemas import ImageEvidence


class VisionAnalyzer:
    def __init__(self, backend: str = config.VISION_BACKEND):
        self.backend = backend
        self._blip_processor = None
        self._blip_model = None
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------ #
    # Lazy model loading (HF backend only)
    # ------------------------------------------------------------------ #
    def _load_blip(self):
        if self._blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self._blip_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL)
        return self._blip_processor, self._blip_model

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL)
            self._clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        return self._clip_model, self._clip_processor

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, image, image_id: str) -> ImageEvidence:
        """`image` is a PIL.Image (already opened by the caller)."""
        if self.backend == "mock":
            return self._analyze_mock(image, image_id)
        try:
            return self._analyze_hf(image, image_id)
        except Exception as e:
            # Never let a missing model / OOM crash the whole assistant --
            # degrade to the mock analyzer and surface the issue in evidence.
            fallback = self._analyze_mock(image, image_id)
            fallback.caption = f"[vision model unavailable, using placeholder analysis: {e}] " + fallback.caption
            return fallback

    # ------------------------------------------------------------------ #
    # HF backend
    # ------------------------------------------------------------------ #
    def _analyze_hf(self, image, image_id: str) -> ImageEvidence:
        import torch

        # --- captioning ---
        proc, model = self._load_blip()
        inputs = proc(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=40)
        caption = proc.decode(out[0], skip_special_tokens=True)

        # --- zero-shot concept tagging + embedding via CLIP ---
        clip_model, clip_proc = self._load_clip()
        clip_inputs = clip_proc(
            text=config.TAG_VOCABULARY, images=image, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            clip_out = clip_model(**clip_inputs)
        probs = clip_out.logits_per_image.softmax(dim=1)[0].tolist()
        tag_scores = dict(zip(config.TAG_VOCABULARY, probs))
        tags = [t for t, s in tag_scores.items() if s >= config.TAG_SCORE_THRESHOLD]
        tags.sort(key=lambda t: -tag_scores[t])

        embedding = clip_out.image_embeds[0].detach().numpy()

        # --- OCR ---
        ocr_text = self._run_ocr(image)

        return ImageEvidence(
            image_id=image_id, caption=caption, tags=tags[:6], tag_scores=tag_scores,
            ocr_text=ocr_text, embedding=embedding, raw_backend="hf",
        )

    def _run_ocr(self, image) -> str:
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).strip()
            return text
        except Exception:
            return ""  # tesseract binary / pytesseract not available -- degrade silently

    # ------------------------------------------------------------------ #
    # Mock backend (offline-friendly, deterministic given the same bytes)
    # ------------------------------------------------------------------ #
    def _analyze_mock(self, image, image_id: str) -> ImageEvidence:
        # Deterministic pseudo-analysis derived from image size/mode so repeated
        # runs on the same image are stable, without needing real model weights.
        w, h = getattr(image, "size", (0, 0))
        mode = getattr(image, "mode", "RGB")
        seed = int(hashlib.md5(f"{w}x{h}{mode}{image_id}".encode()).hexdigest(), 16)

        pseudo_tags = [config.TAG_VOCABULARY[i % len(config.TAG_VOCABULARY)] for i in (seed, seed // 7, seed // 13)]
        pseudo_tags = list(dict.fromkeys(pseudo_tags))  # de-dupe, preserve order
        tag_scores = {t: round(0.2 + (i * 0.15), 2) for i, t in enumerate(pseudo_tags)}

        caption = f"A {w}x{h} image (mock analysis -- no vision model loaded)."
        return ImageEvidence(
            image_id=image_id, caption=caption, tags=pseudo_tags, tag_scores=tag_scores,
            ocr_text="", embedding=None, raw_backend="mock",
        )

    # ------------------------------------------------------------------ #
    # Cross-image similarity (used by context_manager for reference resolution)
    # ------------------------------------------------------------------ #
    def similarity(self, evidence_a: ImageEvidence, evidence_b: ImageEvidence) -> float:
        if evidence_a.embedding is None or evidence_b.embedding is None:
            # Fallback: crude tag-overlap similarity when embeddings aren't available.
            a, b = set(evidence_a.tags), set(evidence_b.tags)
            return len(a & b) / max(len(a | b), 1)
        import numpy as np
        a, b = evidence_a.embedding, evidence_b.embedding
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
