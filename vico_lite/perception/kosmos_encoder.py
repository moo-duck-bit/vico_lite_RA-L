from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class Kosmos2Encoder:
    """Frozen Kosmos-2 vision encoder for optional per-object grounding features.

    Each retained object patch is encoded by the Kosmos-2 vision tower and the
    pooled, L2-normalised feature is attached to the candidate. Nothing is
    trained, and the policy does not read these features.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from transformers import Kosmos2ForConditionalGeneration, Kosmos2Processor  # type: ignore

        self.device = device
        self.processor = Kosmos2Processor.from_pretrained(model_name)
        self.model = Kosmos2ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.requires_grad_(False)
        self.model.eval()
        LOGGER.info("Loaded Kosmos-2 model %s", model_name)

    def encode_images(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        """Encode a batch of RGB patches into (N, dim) L2-normalised features."""
        pixels = self.processor.image_processor(images=list(images), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            pooled = self.model.vision_model(pixel_values=pixels.to(self.device)).pooler_output
        return pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)


__all__ = ["Kosmos2Encoder"]
