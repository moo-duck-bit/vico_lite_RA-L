from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image

from .clip_utils import load_clip

try:
    import clip
except ImportError:  # pragma: no cover - runtime dependency
    clip = None  # type: ignore


def to_uint8_hwc(image) -> np.ndarray:
    """Normalise an RGB tensor/array to an HxWx3 uint8 array."""
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Unexpected image shape {array.shape}")
    if array.shape[0] in (3, 4) and array.shape[2] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = (np.clip(array.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


class FrozenClipEncoder:
    """Frozen CLIP used as a static feature extractor for images and text.

    No parameter of the underlying model is trained or fine-tuned; the encoder
    only exposes L2-normalised embeddings so that a dot product is a cosine
    similarity.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu") -> None:
        self.device = device
        self.model, self.preprocess = load_clip(model_name, device=device)
        self.model.requires_grad_(False)
        self._text_cache: Dict[str, torch.Tensor] = {}

    def encode_images(self, images: Sequence) -> torch.Tensor:
        """Encode a batch of RGB images into (N, dim) L2-normalised embeddings."""
        if not images:
            return torch.empty((0, 0), device=self.device)
        batch = torch.cat(
            [self.preprocess(Image.fromarray(to_uint8_hwc(img))).unsqueeze(0) for img in images],
            dim=0,
        ).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def encode_image(self, image) -> torch.Tensor:
        """Encode a single RGB image into a (dim,) L2-normalised embedding."""
        return self.encode_images([image])[0]

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text into a (dim,) L2-normalised embedding.

        The task instruction is constant within an episode, so embeddings are
        cached per string rather than recomputed every frame.
        """
        if clip is None:
            raise ImportError(
                "The 'clip' package is required. Install via "
                "pip install git+https://github.com/openai/CLIP.git"
            )
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            feature = self.model.encode_text(tokens)
            feature = feature / feature.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        feature = feature.squeeze(0)
        self._text_cache[text] = feature
        return feature


__all__ = ["FrozenClipEncoder", "to_uint8_hwc"]
