from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from ..config import ViCoConfig
from .clip_encoder import FrozenClipEncoder, to_uint8_hwc
from .kosmos_encoder import Kosmos2Encoder
from .regions import extract_object_patches
from .symbolic import build_symbolic_descriptions


@dataclass
class PerceptionOutput:
    """Per-frame output of the grounding pipeline.

    ``agent_summary`` is the mean of the retained object latents and is what the
    team hub pools and smooths; ``symbolic`` is the top-K object table the
    policy actually reads.
    """

    vision_latent: torch.Tensor
    text_latent: torch.Tensor
    agent_summary: torch.Tensor
    symbolic: List[Dict[str, Any]]


class PerceptionAlignmentPipeline:
    """Ground each segmented object against the task instruction with frozen CLIP.

    One image patch is cropped per visible object and encoded by the frozen CLIP
    image encoder; the instruction is encoded once by the frozen text encoder.
    Because both embeddings are L2-normalised, their dot product is the
    object-level relevance score used to rank candidates. No projection head is
    instantiated and no parameter is trained.
    """

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.clip = FrozenClipEncoder(cfg.clip_model, device=device)
        self.kosmos = (
            Kosmos2Encoder(cfg.kosmos_model, device=device, embedding_dim=cfg.kosmos_dim)
            if cfg.kosmos_dim > 0
            else None
        )
        self.output_dim = cfg.latent_dim

    def process(self, observation: Dict[str, Any], instruction: Optional[str]) -> PerceptionOutput:
        rgb = to_uint8_hwc(observation["rgb"])
        text_latent = self.clip.encode_text(instruction or "")
        object_infos = list(observation.get("object_infos") or [])

        patches, indices = extract_object_patches(rgb, observation.get("seg_mask"), object_infos)
        if patches:
            patch_latents = self.clip.encode_images(patches)
            scored_infos = [object_infos[i] for i in indices]
        else:
            # Nothing segmented this frame: fall back to the whole view so the
            # agent still contributes a latent to the team summary.
            patch_latents = self.clip.encode_images([rgb])
            scored_infos = []

        scores = patch_latents @ text_latent
        symbolic = build_symbolic_descriptions(scores, scored_infos, top_k=self.cfg.symbolic_top_k)

        auxiliary = self.kosmos.encode(rgb) if self.kosmos is not None else None
        agent_summary = self._agent_summary(patch_latents, text_latent, auxiliary, symbolic)
        return PerceptionOutput(
            vision_latent=patch_latents.mean(dim=0),
            text_latent=text_latent,
            agent_summary=agent_summary,
            symbolic=symbolic,
        )

    def _agent_summary(
        self,
        patch_latents: torch.Tensor,
        text_latent: torch.Tensor,
        auxiliary: Optional[torch.Tensor],
        symbolic: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """Mean of the retained object latents, each concat(image, text, aux)."""
        retained = len(symbolic[0]["candidates"]) if symbolic else 0
        latents = patch_latents[:retained] if retained else patch_latents
        components = [latents.mean(dim=0), text_latent]
        if auxiliary is not None and auxiliary.numel() > 0:
            components.append(auxiliary)
        summary = torch.cat(components, dim=-1)
        if summary.numel() < self.output_dim:
            padded = torch.zeros(self.output_dim, device=summary.device, dtype=summary.dtype)
            padded[: summary.numel()] = summary
            return padded
        return summary[: self.output_dim]


__all__ = ["PerceptionAlignmentPipeline", "PerceptionOutput"]
