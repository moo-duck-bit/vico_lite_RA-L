from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch


def build_symbolic_descriptions(
    scores: torch.Tensor,
    object_infos: Sequence[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Retain the top-K scored objects as symbolic entries.

    Each entry keeps the object metadata produced by the policy (label,
    estimated 3D position, distance, task flags) and adds the visual-language
    relevance score used to rank it.
    """
    infos = list(object_infos)
    if not infos or scores.numel() == 0:
        return [{"candidates": []}]

    flat = scores.flatten()[: len(infos)]
    k = max(1, min(top_k, flat.numel()))
    values, indices = torch.topk(flat, k=k)

    candidates: List[Dict[str, Any]] = []
    for index, value in zip(indices.tolist(), values.tolist()):
        entry = dict(infos[index])
        entry.setdefault("name", f"obj_{index}")
        entry["score"] = float(value)
        candidates.append(entry)
    return [{"candidates": candidates}]


__all__ = ["build_symbolic_descriptions"]
