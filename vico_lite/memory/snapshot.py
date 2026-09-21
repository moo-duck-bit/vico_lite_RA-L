from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class SharedStateSnapshot:
    """Read-only view of the team memory handed to one agent for one frame."""

    step: int
    ema_latent: torch.Tensor
    history: List[torch.Tensor]
    symbolic: List[Dict[str, Any]]
    per_agent_latents: Dict[int, torch.Tensor] = field(default_factory=dict)
    per_agent_symbolic: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    team_symbolic: List[Dict[str, Any]] = field(default_factory=list)
    agent_id: Optional[int] = None
    skip_targets: Dict[str, List] = field(default_factory=dict)
    nav_guard_info: Dict[Tuple[float, float], int] = field(default_factory=dict)
    obstacle_coords: Dict[Tuple[float, float], int] = field(default_factory=dict)


__all__ = ["SharedStateSnapshot"]
