from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _default_heuristic_weights() -> Dict[str, float]:
    """Weights of the candidate score s(a) = w_d / (d + eps) + (w_n + w_v) * c.

    ``role_alignment`` and ``penalty`` are only consumed by the optional LLM
    branch, which is disabled by default.
    """
    return {
        "distance": 1.5,
        "visibility": 0.6,
        "novelty": 0.8,
        "role_alignment": 0.7,
        "penalty": 1.0,
    }


def _default_reflection_actions() -> List[str]:
    return ["pick", "deliver"]


def _default_action_whitelist() -> Optional[List[str]]:
    return ["search", "move", "pick", "deliver", "assist", "idle"]


@dataclass
class ViCoConfig:
    """Runtime configuration for the ViCo-Lite agent stack.

    The defaults describe the configuration used for every reported
    experiment: training-free, and with zero LLM calls during execution.
    """

    # Frozen visual encoders.
    clip_model: str = "ViT-B/32"
    kosmos_model: str = "microsoft/kosmos-2-patch14-224"
    use_kosmos: bool = False  # Optional Kosmos-2 features on retained candidates; off by default.

    # Number of scored objects retained per frame.
    symbolic_top_k: int = 5

    # Shared memory.
    object_memory_decay: float = 0.5  # Temporal smoothing of stored vs. incoming position/score.

    # Deterministic policy.
    seed: int = 0  # Seeds the fallback-exploration RNG of every agent.
    heuristic_scoring: bool = True  # False falls back to pure distance ranking.
    force_heuristics: bool = True  # True issues no LLM call during execution.

    # Navigation and execution.
    guard_skip_decay: int = 60
    navigation_guard_ratio: float = 1.8
    use_continuous_navigation: bool = True
    continuous_navigation_threshold: float = 2.0  # Metres; below this, use discrete steps.

    # Optional LLM branch. Inactive whenever force_heuristics is True.
    use_reasoner: bool = False
    reasoner_model: str = "gpt-4"
    reasoner_temperature: float = 0.2
    reasoner_candidate_topk: int = 3
    reasoner_min_interval: int = 30  # Frames between two LLM calls.
    reasoner_plan_horizon: int = 15
    reasoner_reflection: bool = False
    reasoner_reflection_interval: int = 5
    reasoner_reflection_actions: List[str] = field(default_factory=_default_reflection_actions)
    reasoner_heuristic_weights: Dict[str, float] = field(default_factory=_default_heuristic_weights)
    action_whitelist: Optional[List[str]] = field(default_factory=_default_action_whitelist)
    action_blacklist: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


__all__ = ["ViCoConfig"]
