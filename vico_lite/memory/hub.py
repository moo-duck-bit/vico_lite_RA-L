from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from ..config import ViCoConfig
from .snapshot import SharedStateSnapshot
from ..perception.pipeline import PerceptionOutput

OBSTACLE_TTL_FRAMES = 120


class TeamMemoryHub:
    """Shared perceptual state for a team of agents.

    Two things are shared. A compact team latent ``m_t`` is the exponential
    moving average of the mean of the per-agent summaries, and an object table
    holds one entry per observed object with its label, smoothed 3D position,
    smoothed relevance score, last-observed step and task status. The object
    table is the interface the policy actually reads; the latent is kept for
    language-based extensions.
    """

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.decay = cfg.memory_ema_decay
        self.object_decay = cfg.object_memory_decay
        self.max_history = cfg.memory_max_history
        self.latent_dim = cfg.latent_dim
        self._agent_latents: Dict[int, torch.Tensor] = {}
        self._agent_symbolic: Dict[int, List[dict]] = {}
        self._agent_skip_names: Dict[int, Set[str]] = {}
        self._agent_skip_coords: Dict[int, Set[Tuple[float, float]]] = {}
        self._agent_skip_ids: Dict[int, Set[int]] = {}
        self._agent_nav_guard: Dict[int, Dict[Tuple[float, float], int]] = {}
        self._agent_obstacles: Dict[int, Dict[Tuple[float, float], int]] = {}
        self._objects: Dict[Any, Dict[str, Any]] = {}
        self._ema: Optional[torch.Tensor] = None
        self._history: Deque[torch.Tensor] = deque(maxlen=self.max_history)
        self._step: int = 0
        self._current_episode: Optional[int] = None

    # -- episode / agent lifecycle -------------------------------------------
    def begin_episode(self, episode_index: Optional[int]) -> None:
        """Clear all shared state when a new episode starts."""
        if episode_index is None:
            episode_index = -1
        if self._current_episode != episode_index:
            self._current_episode = episode_index
            self.reset_episode()

    def reset_episode(self) -> None:
        self._agent_latents.clear()
        self._agent_symbolic.clear()
        self._agent_skip_names.clear()
        self._agent_skip_coords.clear()
        self._agent_skip_ids.clear()
        self._agent_nav_guard.clear()
        self._agent_obstacles.clear()
        self._objects.clear()
        self._ema = None
        self._history.clear()
        self._step = 0

    def register_agent(self, agent_id: int) -> None:
        self._agent_symbolic.setdefault(agent_id, [])
        self._agent_skip_names.setdefault(agent_id, set())
        self._agent_skip_coords.setdefault(agent_id, set())
        self._agent_skip_ids.setdefault(agent_id, set())
        self._agent_nav_guard.setdefault(agent_id, {})
        self._agent_obstacles.setdefault(agent_id, {})

    def reset_agent(self, agent_id: int) -> None:
        self._agent_latents.pop(agent_id, None)
        self._agent_symbolic.pop(agent_id, None)
        self._agent_skip_names.pop(agent_id, None)
        self._agent_skip_coords.pop(agent_id, None)
        self._agent_skip_ids.pop(agent_id, None)
        self._agent_nav_guard.pop(agent_id, None)
        self._agent_obstacles.pop(agent_id, None)

    # -- main update ----------------------------------------------------------
    def update(self, agent_id: int, perception: PerceptionOutput) -> SharedStateSnapshot:
        self.register_agent(agent_id)
        self._update_latent(agent_id, perception.agent_summary)
        self._agent_symbolic[agent_id] = perception.symbolic or []
        self._merge_objects(agent_id, perception.symbolic or [])

        skip_names, skip_coords, skip_ids = self._merged_skip_targets()
        team_symbolic = self._team_objects(skip_names, skip_ids)

        return SharedStateSnapshot(
            step=self._step,
            ema_latent=self._ema.detach().clone()
            if self._ema is not None
            else torch.zeros(self.latent_dim, device=self.device),
            history=list(self._history),
            symbolic=self._agent_symbolic.get(agent_id, []),
            per_agent_latents={idx: tensor.detach().clone() for idx, tensor in self._agent_latents.items()},
            per_agent_symbolic=dict(self._agent_symbolic),
            team_symbolic=team_symbolic,
            agent_id=agent_id,
            skip_targets={
                "names": sorted(skip_names),
                "coords": [list(coord) for coord in sorted(skip_coords)],
                "ids": sorted(skip_ids),
            },
            nav_guard_info=self._merged_nav_guard(),
            obstacle_coords=self._merged_obstacles(),
        )

    def _update_latent(self, agent_id: int, latent: torch.Tensor) -> None:
        """Mean-pool the per-agent summaries, then apply the team-level EMA."""
        latent = latent.detach().to(self.device)
        if latent.ndim > 1:
            latent = latent.flatten()
        self._agent_latents[agent_id] = latent
        mean_latent = torch.stack(list(self._agent_latents.values()), dim=0).mean(dim=0)
        if self._ema is None:
            self._ema = mean_latent
        else:
            self._ema = self.decay * self._ema + (1.0 - self.decay) * mean_latent
        self._history.append(mean_latent.detach())
        self._step += 1

    # -- object table ---------------------------------------------------------
    @staticmethod
    def _object_key(agent_id: int, candidate: Dict[str, Any]) -> Any:
        object_id = candidate.get("id")
        if object_id is not None:
            return ("id", object_id)
        return ("name", str(candidate.get("name", "")).lower(), agent_id)

    def _merge_objects(self, agent_id: int, symbolic: Sequence[Dict[str, Any]]) -> None:
        """Fold this agent's top-K candidates into the shared object table.

        When several agents report the same object, its position and relevance
        score are blended with the stored values instead of overwriting them.
        """
        for row in symbolic:
            for candidate in row.get("candidates", []):
                key = self._object_key(agent_id, candidate)
                entry = self._objects.get(key)
                if entry is None:
                    entry = dict(candidate)
                    entry["observed_by"] = [agent_id]
                else:
                    position = self._blend_position(entry.get("position"), candidate.get("position"))
                    score = self._blend_score(entry.get("score"), candidate.get("score"))
                    entry.update({k: v for k, v in candidate.items() if v is not None})
                    if position is not None:
                        entry["position"] = position
                    if score is not None:
                        entry["score"] = score
                    observers = entry.setdefault("observed_by", [])
                    if agent_id not in observers:
                        observers.append(agent_id)
                entry["last_seen_step"] = self._step
                self._objects[key] = entry

    def _blend_position(self, stored: Any, incoming: Any) -> Optional[List[float]]:
        if incoming is None:
            return list(stored) if isinstance(stored, (list, tuple)) else None
        if not isinstance(stored, (list, tuple)) or len(stored) != len(incoming):
            return list(incoming)
        alpha = self.object_decay
        return [alpha * float(a) + (1.0 - alpha) * float(b) for a, b in zip(stored, incoming)]

    def _blend_score(self, stored: Any, incoming: Any) -> Optional[float]:
        if incoming is None:
            return float(stored) if stored is not None else None
        if stored is None:
            return float(incoming)
        alpha = self.object_decay
        return alpha * float(stored) + (1.0 - alpha) * float(incoming)

    def _team_objects(self, skip_names: Set[str], skip_ids: Set[int]) -> List[Dict[str, Any]]:
        """Flat object list shared with every agent, newest observation first."""
        entries: List[Dict[str, Any]] = []
        for entry in self._objects.values():
            shared = dict(entry)
            name = str(shared.get("name", "")).lower()
            object_id = shared.get("id")
            done = name in skip_names or (object_id is not None and object_id in skip_ids)
            shared["status"] = "done" if done else "pending"
            shared["age"] = self._step - int(shared.get("last_seen_step", self._step))
            entries.append(shared)
        entries.sort(key=lambda item: item.get("last_seen_step", 0), reverse=True)
        return entries

    # -- coordinate sharing ---------------------------------------------------
    @staticmethod
    def _quantize(value: float, step: float = 0.5) -> float:
        if step <= 0:
            return round(value, 2)
        return round(round(value / step) * step, 2)

    def _quantize_coord_pair(self, coord: Sequence[float]) -> Tuple[float, float]:
        return (self._quantize(float(coord[0])), self._quantize(float(coord[1])))

    def _merged_skip_targets(self) -> Tuple[Set[str], Set[Tuple[float, float]], Set[int]]:
        names: Set[str] = set()
        coords: Set[Tuple[float, float]] = set()
        ids: Set[int] = set()
        for values in self._agent_skip_names.values():
            names.update(values)
        for values in self._agent_skip_coords.values():
            coords.update(self._quantize_coord_pair(coord) for coord in values)
        for values in self._agent_skip_ids.values():
            ids.update(values)
        return names, coords, ids

    def _merged_nav_guard(self) -> Dict[Tuple[float, float], int]:
        merged: Dict[Tuple[float, float], int] = {}
        for guard_map in self._agent_nav_guard.values():
            for key, cooldown in guard_map.items():
                coord = self._quantize_coord_pair(key)
                merged[coord] = max(int(cooldown), merged.get(coord, 0))
        return merged

    def _merged_obstacles(self) -> Dict[Tuple[float, float], int]:
        """Union of recently seen obstacles; entries older than the TTL are dropped."""
        merged: Dict[Tuple[float, float], int] = {}
        for agent_id, obstacle_map in self._agent_obstacles.items():
            fresh = {
                coord: frame
                for coord, frame in obstacle_map.items()
                if self._step - frame < OBSTACLE_TTL_FRAMES
            }
            self._agent_obstacles[agent_id] = fresh
            for coord, frame in fresh.items():
                key = self._quantize_coord_pair(coord)
                merged[key] = max(frame, merged.get(key, 0))
        return merged

    def update_skip_targets(
        self,
        agent_id: int,
        names: Iterable[str],
        coords: Iterable[Sequence[float]],
        ids: Iterable[int] = (),
    ) -> None:
        self.register_agent(agent_id)
        self._agent_skip_names[agent_id].update(
            str(name).lower() for name in names if isinstance(name, str)
        )
        for coord in coords:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                self._agent_skip_coords[agent_id].add(self._quantize_coord_pair(coord))
        self._agent_skip_ids[agent_id].update(int(value) for value in ids if value is not None)

    def update_nav_guard(self, agent_id: int, nav_guard: Dict[Sequence[float], int]) -> None:
        self.register_agent(agent_id)
        self._agent_nav_guard[agent_id] = {
            self._quantize_coord_pair(key): int(value)
            for key, value in nav_guard.items()
            if isinstance(key, (list, tuple)) and len(key) >= 2
        }

    def update_obstacles(
        self,
        agent_id: int,
        obstacle_coords: Iterable[Sequence[float]],
        current_frame: int,
    ) -> None:
        self.register_agent(agent_id)
        store = self._agent_obstacles[agent_id]
        for coord in obstacle_coords:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                key = self._quantize_coord_pair(coord)
                store[key] = max(current_frame, store.get(key, 0))


__all__ = ["TeamMemoryHub"]
