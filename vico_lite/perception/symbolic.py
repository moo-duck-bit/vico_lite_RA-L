from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


def build_symbolic_descriptions(
    scores: torch.Tensor,
    object_infos: Sequence[Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Retain the min(K, N_vis) highest-scored objects as symbolic entries.

    Each entry keeps the object metadata from the perception frontend (label,
    estimated 3D position, distance, task flags) and adds the CLIP relevance
    score. Also returns the index into ``object_infos`` of every retained entry.
    """
    infos = list(object_infos)
    if not infos or scores.numel() == 0:
        return [{"candidates": []}], []

    flat = scores.flatten()[: len(infos)]
    k = max(1, min(top_k, flat.numel()))
    values, indices = torch.topk(flat, k=k)

    candidates: List[Dict[str, Any]] = []
    for index, value in zip(indices.tolist(), values.tolist()):
        entry = dict(infos[index])
        entry.setdefault("name", f"obj_{index}")
        entry["score"] = float(value)
        candidates.append(entry)
    return [{"candidates": candidates}], indices.tolist()


__all__ = ["build_symbolic_descriptions"]
