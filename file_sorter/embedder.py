from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .scanner import FileInfo

logger = logging.getLogger(__name__)

# CLIP text encoder accepts up to ~77 tokens; ~2000 chars is a safe limit
_MAX_TEXT_CHARS = 2000


def _extract_text(info: FileInfo) -> Optional[str]:
    """Extract meaningful text from a file to feed into the text encoder."""
    try:
        if info.file_type in ('text', 'code'):
            return info.path.read_text(errors='replace')[:_MAX_TEXT_CHARS]

        if info.extension == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(info.path))
                pages_text = []
                for page in reader.pages[:5]:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                return ' '.join(pages_text)[:_MAX_TEXT_CHARS] or None
            except Exception:
                return None

        if info.extension == '.docx':
            try:
                from docx import Document
                doc = Document(str(info.path))
                text = ' '.join(p.text for p in doc.paragraphs[:30])
                return text[:_MAX_TEXT_CHARS] or None
            except Exception:
                return None

    except Exception as e:
        logger.debug("Text extraction failed for %s: %s", info.path, e)

    return None


def _filename_description(info: FileInfo) -> str:
    """Human-readable description of a file from its name and location."""
    stem = info.stem.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    parent = info.path.parent.name
    parts = [stem]
    if parent and parent not in ('.', '..', 'Desktop', 'Downloads'):
        parts.append(f"in {parent.replace('_', ' ').replace('-', ' ')}")
    parts.append(info.file_type)
    return ' '.join(parts)


class Embedder:
    """Generates CLIP embeddings for files. Images go through the vision encoder;
    text files go through the text encoder — both land in the same 512-dim space."""

    def __init__(self, model_name: str = 'clip-ViT-B-32', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_image(self, path: Path) -> Optional[np.ndarray]:
        try:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            return self.model.encode(img, normalize_embeddings=True)
        except Exception as e:
            logger.debug("Image embed failed for %s: %s", path, e)
            return None

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_video_keyframe(self, path: Path) -> Optional[np.ndarray]:
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(1, frame_count // 10))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return self.model.encode(img, normalize_embeddings=True)
        except Exception as e:
            logger.debug("Video embed failed for %s: %s", path, e)
            return None

    def embed_file(self, info: FileInfo) -> Optional[np.ndarray]:
        if info.file_type == 'image':
            emb = self.embed_image(info.path)
            if emb is not None:
                return emb

        if info.file_type == 'video':
            emb = self.embed_video_keyframe(info.path)
            if emb is not None:
                return emb

        if info.file_type in ('document', 'text', 'code'):
            text = _extract_text(info)
            if text:
                return self.embed_text(text)

        # Fallback: embed a descriptive string derived from filename + folder context
        return self.embed_text(_filename_description(info))
