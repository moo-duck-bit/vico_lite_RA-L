from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from agent_memory import AgentMemory
from vico_lite.config import ViCoConfig
from vico_lite.memory import TeamMemoryHub
from vico_lite.policy import ViCoPolicy

MAP_SIZE = (240, 120)
SCENE_BOUNDS = {"x_min": -15.0, "x_max": 15.0, "z_min": -7.5, "z_max": 7.5}


class ViCoAgent:
    """Gym-side wrapper binding one replicant to a ViCo-Lite policy.

    ``memory_hub`` is shared by every agent of a team. Handing each agent its
    own hub instead reproduces the ablation in which shared memory is removed.
    """

    agent_type = "vico_agent"

    def __init__(
        self,
        agent_id: int,
        logger,
        max_frames: int,
        cfg: Optional[ViCoConfig] = None,
        memory_hub: Optional[TeamMemoryHub] = None,
        output_dir: str = "results",
        device: str = "cpu",
    ) -> None:
        self.agent_id = agent_id
        self.logger = logger
        self.max_frames = max_frames
        self.cfg = cfg or ViCoConfig()
        self.memory_hub = memory_hub or TeamMemoryHub(self.cfg, device=device)
        self.output_dir = output_dir
        self.device = device
        self.save_img = True
        self.gt_mask = True
        self.goal_objects: Dict[str, Any] = {}
        self.rooms_name = None
        self.agent_memory: Optional[AgentMemory] = None
        self.policy: Optional[ViCoPolicy] = None
        self.env_api: Optional[Dict[str, Any]] = None
        self.agent_color = [-1, -1, -1]

    def reset(
        self,
        obs: Dict[str, Any],
        goal_objects: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        env_api: Optional[Any] = None,
        agent_color=None,
        agent_id: int = 0,
        rooms_name=None,
        gt_mask: bool = True,
        save_img: bool = True,
        episode_index: Optional[int] = None,
    ) -> None:
        self.output_dir = output_dir or self.output_dir
        self.goal_objects = goal_objects or {}
        self.rooms_name = rooms_name
        self.gt_mask = gt_mask
        self.save_img = save_img
        self.agent_color = agent_color if agent_color is not None else [-1, -1, -1]
        self.env_api = env_api[self.agent_id] if isinstance(env_api, list) else (env_api or {})

        self.agent_memory = AgentMemory(
            agent_id=self.agent_id,
            agent_color=self.agent_color,
            output_dir=self.output_dir,
            gt_mask=self.gt_mask,
            gt_behavior=True,
            env_api=self.env_api,
            constraint_type=None,
            map_size=MAP_SIZE,
            scene_bounds=SCENE_BOUNDS,
        )
        self.agent_memory.logger = self.logger

        self.memory_hub.begin_episode(episode_index)
        self.memory_hub.register_agent(self.agent_id)
        self.policy = ViCoPolicy(
            self.cfg,
            agent_id=self.agent_id,
            logger=self.logger,
            agent_memory=self.agent_memory,
            memory_hub=self.memory_hub,
            env_api=self.env_api,
            device=self.device,
        )
        self.policy.reset()
        self.policy.set_goal_context(goal_objects=self.goal_objects, rooms_name=self.rooms_name)
        self.agent_memory.update(self._ground_observation(obs), save_img=self.save_img)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.agent_memory is None or self.policy is None:
            raise RuntimeError("Agent not initialised. Call reset() first.")
        obs = self._ground_observation(obs)
        self.agent_memory.update(obs, save_img=self.save_img)
        return self.policy.act(self._augment_observation(obs))

    # ------------------------------------------------------------------
    def _ground_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Recover segmentation under non-oracle perception.

        With ground-truth masks the environment already supplies
        ``visible_objects`` and ``seg_mask``. Without them, labels and masks are
        estimated from the egocentric RGB frame by the same detector the
        baselines use, so both settings hand the policy the same interface.
        """
        if self.gt_mask or self.agent_memory is None:
            return obs
        obs = dict(obs)
        # The detector expects HxWx3; the environment hands out 3xHxW.
        obs["visible_objects"], obs["seg_mask"] = self.agent_memory.detect(
            obs["rgb"].transpose(1, 2, 0)
        )
        return obs

    def _augment_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        augmented = copy.deepcopy(obs)
        augmented.setdefault("instruction", self._goal_instruction())
        augmented.setdefault("visible_objects", [])
        return augmented

    def _goal_instruction(self) -> str:
        if not self.goal_objects:
            return "goal: cooperate to transport target objects"
        parts = [f"{count} x {name}" for name, count in self.goal_objects.items()]
        return "Collect " + ", ".join(parts)
