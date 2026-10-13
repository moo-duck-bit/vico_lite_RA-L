from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import ViCoConfig
from .clip_encoder import FrozenClipEncoder, to_uint8_hwc
from .regions import extract_object_patches
from .symbolic import build_symbolic_descriptions


@dataclass
class PerceptionOutput:
    """Per-frame output of the grounding pipeline: the top-K object entries."""

    symbolic: List[Dict[str, Any]]


class PerceptionAlignmentPipeline:
    """Ground each segmented object against the task instruction with frozen CLIP.

    One image patch is cropped per visible object and encoded by the frozen CLIP
    image encoder; the instruction is encoded once by the frozen text encoder.
    Both embeddings are L2-normalised, so their dot product c_j (Eq. 2) is the
    object-level relevance score. No projection head is instantiated and no
    parameter is trained.
    """

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.clip = FrozenClipEncoder(cfg.clip_model, device=device)
        self.kosmos = None
        if cfg.use_kosmos:
            from .kosmos_encoder import Kosmos2Encoder

            self.kosmos = Kosmos2Encoder(cfg.kosmos_model, device=device)

    def process(self, observation: Dict[str, Any], instruction: Optional[str]) -> PerceptionOutput:
        object_infos = list(observation.get("object_infos") or [])
        rgb = to_uint8_hwc(observation["rgb"])
        patches, indices = extract_object_patches(rgb, observation.get("seg_mask"), object_infos)
        if not patches:
            return PerceptionOutput(symbolic=[{"candidates": []}])

        text_latent = self.clip.encode_text(instruction or "")
        scores = self.clip.encode_images(patches) @ text_latent
        scored_infos = [object_infos[i] for i in indices]
        symbolic, retained = build_symbolic_descriptions(scores, scored_infos, top_k=self.cfg.symbolic_top_k)

        if self.kosmos is not None and retained:
            features = self.kosmos.encode_images([patches[i] for i in retained])
            for candidate, feature in zip(symbolic[0]["candidates"], features):
                candidate["kosmos_feature"] = feature.cpu().tolist()
        return PerceptionOutput(symbolic=symbolic)


__all__ = ["PerceptionAlignmentPipeline", "PerceptionOutput"]
