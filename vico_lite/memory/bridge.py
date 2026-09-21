from __future__ import annotations

from typing import Any, Dict, List

from ..config import ViCoConfig
from ..perception.pipeline import PerceptionAlignmentPipeline, PerceptionOutput
from .action_recorder import ActionRecorder
from .hub import TeamMemoryHub
from .snapshot import SharedStateSnapshot


class MemoryBridgeController:
    """Per-agent glue between the perception pipeline and the team memory.

    Each frame it grounds the observation, publishes the result to the hub and
    hands back the resulting shared snapshot. With ``--no_shared_memory`` each
    agent is given its own hub, so this class is unchanged between the full
    system and the ablation.
    """

    def __init__(
        self,
        cfg: ViCoConfig,
        agent_id: int,
        hub: TeamMemoryHub,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.device = device
        self.hub = hub
        self.action_recorder = ActionRecorder()
        self.perception = PerceptionAlignmentPipeline(cfg, device=device)

    def reset(self) -> None:
        self.action_recorder.clear()
        self.hub.register_agent(self.agent_id)

    def process(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        perception: PerceptionOutput = self.perception.process(
            observation, observation.get("instruction")
        )
        snapshot: SharedStateSnapshot = self.hub.update(self.agent_id, perception)
        return {"perception": perception, "snapshot": snapshot}

    def record_action(self, data: Dict[str, Any]) -> None:
        self.action_recorder.record(data)

    def recent_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.action_recorder.last(n)


__all__ = ["MemoryBridgeController"]
