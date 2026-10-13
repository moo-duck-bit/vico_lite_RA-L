from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..config import ViCoConfig
from .snapshot import SharedStateSnapshot
from ..perception.pipeline import PerceptionOutput

OBSTACLE_TTL_FRAMES = 120
GRID_RESOLUTION = 0.5  # Metres; spatial keys of the shared memory are quantised to this grid.


def quantize(value: float, step: float = GRID_RESOLUTION) -> float:
    return round(round(value / step) * step, 2)


def quantize_xz(coord: Sequence[float]) -> Tuple[float, float]:
    """Grid cell of a coordinate: (x, z) for 3D positions, (x, y) for 2D pairs."""
    if len(coord) >= 3:
        return (quantize(float(coord[0])), quantize(float(coord[2])))
    return (quantize(float(coord[0])), quantize(float(coord[1])))


class TeamMemoryHub:
    """Shared object-level memory for a team of agents (Sec. III-C).

    For each object o_j the table stores its label l_j, 3D position x_j,
    relevance score c_j, last-observed timestep tau_j and task status q_j.
    Observations of the same object from several agents are temporally
    smoothed rather than overwritten.

    Two update paths are kept separate, as in the efficiency analysis:

    * ``update`` (the shared-memory update) runs once per agent per step and
      folds that agent's top-K candidates into the object table;
    * ``_sync_team`` (the team hub update) runs once per environment step and
      merges the execution feedback of all agents -- skip targets, navigation
      guards and obstacles -- into the task status and the shared views.
    """

    def __init__(self, cfg: ViCoConfig, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self.object_decay = cfg.object_memory_decay
        self._agent_symbolic: Dict[int, List[dict]] = {}
        self._agent_skip_names: Dict[int, Set[str]] = {}
        self._agent_skip_coords: Dict[int, Set[Tuple[float, float]]] = {}
        self._agent_skip_ids: Dict[int, Set[int]] = {}
        self._agent_nav_guard: Dict[int, Dict[Tuple[float, float], int]] = {}
        self._agent_obstacles: Dict[int, Dict[Tuple[float, float], int]] = {}
        self._objects: Dict[Any, Dict[str, Any]] = {}
        self._frame: int = 0
        self._synced_frame: Optional[int] = None
        self._team_skip: Tuple[Set[str], Set[Tuple[float, float]], Set[int]] = (set(), set(), set())
        self._team_nav_guard: Dict[Tuple[float, float], int] = {}
        self._team_obstacles: Dict[Tuple[float, float], int] = {}
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
        self._agent_symbolic.clear()
        self._agent_skip_names.clear()
        self._agent_skip_coords.clear()
        self._agent_skip_ids.clear()
        self._agent_nav_guard.clear()
        self._agent_obstacles.clear()
        self._objects.clear()
        self._frame = 0
        self._synced_frame = None
        self._team_skip = (set(), set(), set())
        self._team_nav_guard = {}
        self._team_obstacles = {}

    def register_agent(self, agent_id: int) -> None:
        self._agent_symbolic.setdefault(agent_id, [])
        self._agent_skip_names.setdefault(agent_id, set())
        self._agent_skip_coords.setdefault(agent_id, set())
        self._agent_skip_ids.setdefault(agent_id, set())
        self._agent_nav_guard.setdefault(agent_id, {})
        self._agent_obstacles.setdefault(agent_id, {})

    def reset_agent(self, agent_id: int) -> None:
        self._agent_symbolic.pop(agent_id, None)
        self._agent_skip_names.pop(agent_id, None)
        self._agent_skip_coords.pop(agent_id, None)
        self._agent_skip_ids.pop(agent_id, None)
        self._agent_nav_guard.pop(agent_id, None)
        self._agent_obstacles.pop(agent_id, None)

    # -- main update ----------------------------------------------------------
    def update(self, agent_id: int, perception: PerceptionOutput, frame: int) -> SharedStateSnapshot:
        """Shared-memory update for one agent at environment step ``frame``."""
        self.register_agent(agent_id)
        self._frame = int(frame)
        if self._synced_frame != self._frame:
            self._sync_team()
        self._agent_symbolic[agent_id] = perception.symbolic or []
        self._merge_objects(agent_id, perception.symbolic or [])

        skip_names, skip_coords, skip_ids = self._team_skip
        return SharedStateSnapshot(
            step=self._frame,
            symbolic=self._agent_symbolic.get(agent_id, []),
            per_agent_symbolic=dict(self._agent_symbolic),
            team_symbolic=self._team_objects(),
            agent_id=agent_id,
            skip_targets={
                "names": sorted(skip_names),
                "coords": [list(coord) for coord in sorted(skip_coords)],
                "ids": sorted(skip_ids),
            },
            nav_guard_info=dict(self._team_nav_guard),
            obstacle_coords=dict(self._team_obstacles),
        )

    def _sync_team(self) -> None:
        """Team hub update: merge every agent's execution feedback, once per step."""
        names: Set[str] = set()
        coords: Set[Tuple[float, float]] = set()
        ids: Set[int] = set()
        for values in self._agent_skip_names.values():
            names.update(values)
        for values in self._agent_skip_coords.values():
            coords.update(values)
        for values in self._agent_skip_ids.values():
            ids.update(values)
        self._team_skip = (names, coords, ids)

        nav_guard: Dict[Tuple[float, float], int] = {}
        for guard_map in self._agent_nav_guard.values():
            for coord, cooldown in guard_map.items():
                nav_guard[coord] = max(int(cooldown), nav_guard.get(coord, 0))
        self._team_nav_guard = nav_guard

        obstacles: Dict[Tuple[float, float], int] = {}
        for agent_id, obstacle_map in self._agent_obstacles.items():
            fresh = {
                coord: seen for coord, seen in obstacle_map.items()
                if self._frame - seen < OBSTACLE_TTL_FRAMES
            }
            self._agent_obstacles[agent_id] = fresh
            for coord, seen in fresh.items():
                obstacles[coord] = max(seen, obstacles.get(coord, 0))
        self._team_obstacles = obstacles

        # Task status q_j: an object is done once any agent has delivered it or
        # given up on it. Status is tracked by object id only, so delivering one
        # apple never marks a second apple as done.
        for entry in self._objects.values():
            if entry.get("id") in ids:
                entry["status"] = "done"
        self._synced_frame = self._frame

    # -- object table ---------------------------------------------------------
    @staticmethod
    def _object_key(candidate: Dict[str, Any]) -> Any:
        object_id = candidate.get("id")
        if object_id is not None:
            return ("id", object_id)
        # Without an id, the same label in the same 0.5 m cell is the same object,
        # whichever agent reports it.
        name = str(candidate.get("name", "")).lower()
        position = candidate.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            return ("cell", name, quantize_xz(position))
        return ("name", name)

    def _merge_objects(self, agent_id: int, symbolic: Sequence[Dict[str, Any]]) -> None:
        """Fold this agent's top-K candidates into the shared object table."""
        for row in symbolic:
            for candidate in row.get("candidates", []):
                key = self._object_key(candidate)
                entry = self._objects.get(key)
                if entry is None:
                    entry = dict(candidate)
                    entry["observed_by"] = [agent_id]
                    entry["status"] = "pending"
                else:
                    position = self._blend_position(entry.get("position"), candidate.get("position"))
                    score = self._blend_score(entry.get("score"), candidate.get("score"))
                    status = entry.get("status", "pending")
                    entry.update({k: v for k, v in candidate.items() if v is not None})
                    entry["status"] = status
                    if position is not None:
                        entry["position"] = position
                    if score is not None:
                        entry["score"] = score
                    if agent_id not in entry["observed_by"]:
                        entry["observed_by"].append(agent_id)
                if entry.get("id") in self._team_skip[2]:
                    entry["status"] = "done"
                entry["last_seen_step"] = self._frame
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

    def _team_objects(self) -> List[Dict[str, Any]]:
        """Flat object list shared with every agent, newest observation first."""
        entries = []
        for entry in self._objects.values():
            shared = dict(entry)
            shared["age"] = self._frame - int(shared.get("last_seen_step", self._frame))
            entries.append(shared)
        entries.sort(key=lambda item: item.get("last_seen_step", 0), reverse=True)
        return entries

    # -- execution feedback ---------------------------------------------------
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
                self._agent_skip_coords[agent_id].add(quantize_xz(coord))
        self._agent_skip_ids[agent_id].update(int(value) for value in ids if value is not None)

    def update_nav_guard(self, agent_id: int, nav_guard: Dict[Sequence[float], int]) -> None:
        self.register_agent(agent_id)
        self._agent_nav_guard[agent_id] = {
            quantize_xz(key): int(value)
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
                key = quantize_xz(coord)
                store[key] = max(current_frame, store.get(key, 0))


__all__ = ["GRID_RESOLUTION", "OBSTACLE_TTL_FRAMES", "TeamMemoryHub", "quantize", "quantize_xz"]
