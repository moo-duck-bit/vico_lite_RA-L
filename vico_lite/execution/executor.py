from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..config import ViCoConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    plan: Dict[str, Any]
    navigation: Dict[str, Any]
    forced_failure: bool = False


class PlanExecutor:
    def __init__(
        self,
        cfg: ViCoConfig,
        agent_id: int,
        agent_memory,
        env_api=None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        self.agent_memory = agent_memory
        self.env_api = env_api
        self.logger = logger or LOGGER
        self.guard_skip_decay = cfg.guard_skip_decay
        self._guard_skip: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # ---------------------------------------------------------------------
    def execute(self, plan, snapshot, agent_state) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        action_type = plan.action_type
        meta: Dict[str, Any] = {"plan": plan.meta}
        command: Dict[str, Any]
        if action_type == "deliver":
            # Deliver: navigate to target, then put_in
            command, nav_meta = self._navigate_and_deliver(plan, agent_state)
            meta.update(nav_meta)
        elif action_type in {"move", "search", "assist"}:
            command, nav_meta = self._navigate_to(plan.target_position, agent_state, follow=False)
            meta.update(nav_meta)
        elif action_type == "pick":
            command, nav_meta = self._approach_and_pick(plan, agent_state)
            meta.update(nav_meta)
        elif action_type in {"idle", "wait"}:
            command = {"type": "ongoing"}
        else:
            command = {"type": "ongoing"}
            meta["unknown_action"] = action_type
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug(
                "[Executor] grounded plan=%s -> command=%s meta=%s",
                plan,
                command,
                meta,
            )
        return command, meta

    # ---------------------------------------------------------------------
    def _navigate_and_deliver(self, plan, agent_state):
        """Navigate to target container and execute put_in action."""
        target_pos = plan.target_position
        target_id = plan.target_id
        meta = {"navigation": "deliver"}
        
        if target_pos is None:
            if self.logger:
                self.logger.warning("[Executor] deliver: target_pos is None")
            meta.update({"reason": "no_target_position"})
            return {"type": "ongoing"}, meta
        
        # The distance decides the navigation method, so measure it first.
        # This ensures we can detect when agent is close enough for put_in even during continuous navigation
        # Calculate actual distance from agent to target
        actual_distance = float("inf")
        goal_position_id = None
        if self.env_api is not None:
            if isinstance(self.env_api, dict):
                goal_position_id = self.env_api.get("goal_position_id")
            elif hasattr(self.env_api, "goal_position_id"):
                goal_position_id = self.env_api.goal_position_id
        
        # Check if target_id is goal_position_id - if so, we need to check distance to goal_position_id
        is_goal_position = (goal_position_id is not None and target_id == goal_position_id) or plan.meta.get("is_goal_position", False)
        
        if self.agent_memory is not None and hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    
                    # If this is goal_position_id, try to get actual goal_position_id position
                    if is_goal_position and goal_position_id is not None:
                        # Try to get goal_position_id position from agent_memory or env_api
                        goal_pos = None
                        if hasattr(self.agent_memory, "object_info") and goal_position_id in self.agent_memory.object_info:
                            goal_pos = self.agent_memory.object_info[goal_position_id].get("position")
                        if goal_pos is None and hasattr(self.agent_memory, "get_object_list"):
                            # Try to get from get_object_list which has goal_position_id position
                            try:
                                obj_list = self.agent_memory.get_object_list()
                                for obj in obj_list:
                                    if obj.get("id") == goal_position_id:
                                        goal_pos = obj.get("position")
                                        break
                            except Exception:
                                pass
                        
                        if goal_pos is not None:
                            if isinstance(goal_pos, (list, tuple, np.ndarray)) and len(goal_pos) >= 3:
                                goal_arr = np.array([goal_pos[0], goal_pos[2]])
                                actual_distance = float(np.linalg.norm(agent_arr - goal_arr))
                        else:
                            # Fallback to target_pos if goal_position_id position not available
                            target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                            actual_distance = float(np.linalg.norm(agent_arr - target_arr))
                    else:
                        # Not goal_position_id, use target_pos
                        target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                        actual_distance = float(np.linalg.norm(agent_arr - target_arr))
            except Exception as exc:
                if self.logger:
                    self.logger.debug("[Executor] deliver: failed to calculate actual distance: %s", exc)
        
        # Also check distance from plan.meta if available (this is the distance calculated in policy)
        plan_distance = plan.meta.get("distance")
        if plan_distance is not None and isinstance(plan_distance, (int, float)):
            # Use the smaller of actual_distance and plan_distance
            if actual_distance == float("inf") or plan_distance < actual_distance:
                actual_distance = plan_distance
        
        # If we're close enough (actual distance, not path_len), execute put_in IMMEDIATELY
        # TDW PutIn requires agent to be close to goal_position_id (must be < 3.0m for success)
        # For goal_position_id, use 2.5m threshold to ensure we're well within 3.0m limit
        # For other containers, use 2.0m threshold
        distance_threshold = 2.5 if is_goal_position else 2.0
        if actual_distance is not None and actual_distance <= distance_threshold:
            # Close enough to put_in - execute immediately
            if self.logger:
                self.logger.info(
                    "[Executor] deliver: close enough (actual_dist=%.2f <= %.2f), executing put_in immediately",
                    actual_distance,
                    distance_threshold,
                )
            # Execute put_in action (type 4)
            command = {"type": 4}
            meta.update({"result": "put_in", "distance": actual_distance, "actual_distance": actual_distance})
            return command, meta
        
        # Not close enough yet - navigate to target first
        # For deliver actions, use continuous navigation if distance > 3.0m to avoid guard blocking
        # This ensures we can get close enough for put_in to succeed
        use_continuous_for_deliver = False
        if actual_distance != float("inf"):
            # Use continuous navigation for deliver if distance > 3.0m (to avoid guard blocking)
            # This is lower than the normal 5.0m threshold to ensure deliver can proceed
            use_continuous_for_deliver = actual_distance > 3.0
        
        if use_continuous_for_deliver and self.cfg.use_continuous_navigation:
            # Force continuous navigation for deliver to avoid guard blocking
            # Use arrived_at (2.5m) for deliver to ensure we get close enough for put_in
            arrived_at = 2.5  # put_in succeeds from 2.46 m, so stopping earlier wastes frames.
            action = {
                "type": 9,  # Continuous movement
                "target_position": target_pos,
                "arrived_at": arrived_at,  # Explicitly set arrived_at for deliver
            }
            nav_meta = {
                "navigation": "continuous_move_to_position",
                "distance": actual_distance,
                "method": "continuous",
                "arrived_at": arrived_at,
            }
            if self.logger:
                self.logger.debug(
                    "[Executor] deliver: using continuous navigation (dist=%.2f > 3.0m, arrived_at=%.2fm)",
                    actual_distance,
                    arrived_at,
                )
            command = action
            meta.update(nav_meta)
        else:
            command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, is_goal_position=is_goal_position)
            meta.update(nav_meta)
            # Check if navigation failed but we're still close enough
            # Even when navigation reports an obstacle, try put_in if close enough.
            if nav_meta.get("forced_failure") and actual_distance <= distance_threshold:
                if self.logger:
                    self.logger.info(
                        "[Executor] deliver: navigation failed but close enough (dist=%.2f <= %.2f), attempting put_in",
                        actual_distance,
                        distance_threshold,
                    )
                command = {"type": 4}
                meta.update({"result": "put_in_after_nav_failure", "distance": actual_distance, "distance_threshold": distance_threshold})
            elif nav_meta.get("forced_failure") and actual_distance != float("inf"):
                # Relax the threshold slightly on repeated delivery failures,
                # since the remaining gap is usually a blocked last metre.
                deliver_fail_count = agent_state.get("deliver_fail_count", {}).get(target_id, 0)
                dynamic_threshold = min(distance_threshold + (deliver_fail_count * 0.1), 3.0)
                if actual_distance <= dynamic_threshold:
                    if self.logger:
                        self.logger.info(
                            "[Executor] deliver: navigation failed but within dynamic threshold (dist=%.2f <= %.2f, fail_count=%d), attempting put_in",
                            actual_distance,
                            dynamic_threshold,
                            deliver_fail_count,
                        )
                    command = {"type": 4}
                    meta.update({"result": "put_in_after_nav_failure_dynamic", "distance": actual_distance, "dynamic_threshold": dynamic_threshold, "deliver_fail_count": deliver_fail_count})
        
        return command, meta

    # ---------------------------------------------------------------------
    def _navigate_to(self, target_pos, agent_state, follow: bool = False, skip_clamp: bool = False, arrived_at: Optional[float] = None, is_goal_position: bool = False):
        meta: Dict[str, Any] = {"navigation": "idle"}
        if target_pos is None:
            return {"type": "ongoing"}, meta
        if self.agent_memory is None:
            meta["reason"] = "no_agent_memory"
            return {"type": "ongoing"}, meta
        
        # Pick actions keep the raw object position; everything else is clamped
        # into the floorplan first.
        if not skip_clamp:
            target_pos = self._clamp_target(target_pos)
        
        # Final validation: if target is still outside room after clamping, reject it
        # BUT: Skip validation for pick actions (skip_clamp=True) - they need actual object position
        if not skip_clamp and self.env_api and "check_pos_in_room" in self.env_api:
            if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                # Target is still outside room, use agent's current position
                if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
                    try:
                        agent_pos = self.agent_memory.position
                        if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                            target_pos = (float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2]))
                            meta["reason"] = "target_outside_room_using_agent_pos"
                            # Verify agent position is valid
                            if not self.env_api["check_pos_in_room"]((target_pos[0], target_pos[2])):
                                meta["reason"] = "target_outside_room_agent_pos_invalid"
                                return {"type": "ongoing"}, meta
                        else:
                            meta["reason"] = "target_outside_room_invalid"
                            return {"type": "ongoing"}, meta
                    except Exception:
                        meta["reason"] = "target_outside_room_error"
                        return {"type": "ongoing"}, meta
                else:
                    meta["reason"] = "target_outside_room_no_agent_pos"
                    return {"type": "ongoing"}, meta
        
        if hasattr(self.agent_memory, "belongs_to_which_room"):
            room = self.agent_memory.belongs_to_which_room(target_pos)
            if room is None and hasattr(self.agent_memory, "center_of_room"):
                current_room = agent_state.get("current_room")
                if current_room is not None:
                    room_center = self.agent_memory.center_of_room(current_room)
                    if room_center is not None:
                        target_pos = (room_center[0], target_pos[1], room_center[2])
        
        # Continuous movement for long distances, discrete steps for short ones.
        # Calculate distance to target
        distance_to_target = float("inf")
        if hasattr(self.agent_memory, "position") and self.agent_memory.position is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                    distance_to_target = float(np.linalg.norm(agent_arr - target_arr))
            except Exception:
                pass
        
        # Use continuous movement for long distances if enabled
        if self.cfg.use_continuous_navigation and distance_to_target > self.cfg.continuous_navigation_threshold:
            # Use continuous movement (type 9: move_to_position)
            if self.logger:
                self.logger.debug(
                    "[Executor] using continuous navigation for long distance: %.2fm > %.2fm, target=%s",
                    distance_to_target,
                    self.cfg.continuous_navigation_threshold,
                    target_pos,
                )
            # Return continuous movement action (type 9)
            # Use smaller arrived_at for pick actions (0.3m) to get closer to objects
            # Stopping a little short of the target saves frames without losing reach.
            if arrived_at is None:
                arrived_at = 1.0
            action = {
                "type": 9,  # New action type for continuous movement
                "target_position": target_pos,
                "arrived_at": arrived_at,  # Can be 0.3m for pick, 0.7m for general navigation
            }
            meta.update({
                "navigation": "continuous_move_to_position",
                "distance": distance_to_target,
                "method": "continuous",
                "arrived_at": arrived_at,
            })
            return action, meta
        else:
            # Discrete step-by-step movement, as in the CoELA baseline.
            action, path_len = self.agent_memory.move_to_pos(target_pos, follow=follow)
            meta.update({
                "navigation": "move_to_pos",
                "distance": path_len,
                "method": "discrete",
            })
            guard_threshold = self._guard_threshold()
            # Skip implausibly long paths, but allow twice the budget for a
            # delivery, which has to reach the goal to score at all.
            max_path_length = 200 if is_goal_position else 100
            if path_len is not None and path_len > max_path_length:
                key = tuple(int(x * 10) for x in target_pos[:2])
                self._guard_skip[key] = {"frames": self.guard_skip_decay, "target": target_pos}
                meta.update({"navigation": "guard_turn", "distance": path_len, "reason": "path_too_long", "forced_failure": True})
                if self.logger:
                    self.logger.warning(
                        "[Executor] Guard triggered for agent %s (len=%.2f > max=%d) target=%s - path too long, skipping (is_goal_position=%s)",
                        self.agent_id,
                        path_len,
                        max_path_length,
                        target_pos,
                        is_goal_position,
                    )
                return {"type": "ongoing"}, meta
            elif path_len is not None and guard_threshold is not None and path_len > guard_threshold:
                key = tuple(int(x * 10) for x in target_pos[:2])
                self._guard_skip[key] = {"frames": self.guard_skip_decay, "target": target_pos}
                meta.update({"navigation": "guard_turn", "distance": path_len, "reason": "path_too_long", "forced_failure": True})
                if self.logger:
                    self.logger.warning(
                        "[Executor] Guard triggered for agent %s (len=%.2f thr=%.2f) target=%s",
                        self.agent_id,
                        path_len,
                        guard_threshold,
                        target_pos,
                    )
                return {"type": "ongoing"}, meta
            if self.logger and getattr(self.logger, "debug", None):
                self.logger.debug(
                    "[Executor] navigate target=%s action=%s path_len=%s",
                    target_pos,
                    action,
                    path_len,
                )
            return action or {"type": "ongoing"}, meta

    def _approach_and_pick(self, plan, agent_state):
        target_pos = plan.target_position
        meta = {"navigation": "pick"}
        is_force_pick = plan.meta.get("override") == "near_grabbable" or plan.meta.get("reason") == "policy_override_grabbable"
        force_pick_distance = plan.meta.get("distance")
        
        # A degenerate (0,0,0) target is still actionable: reach_for only needs
        # the object id, so fall back to it rather than rejecting the plan.
        target_id = plan.target_id
        is_task_target = plan.meta.get("is_task_target", False)
        has_invalid_position = target_pos is not None and (target_pos == (0.0, 0.0, 0.0) or (abs(target_pos[0]) < 0.01 and abs(target_pos[2]) < 0.01))
        
        if has_invalid_position:
            if target_id is not None and (is_task_target or is_force_pick):
                # Known task target: pick by id alone.
                if self.logger:
                    self.logger.info(
                        "[Executor] pick: invalid target_pos (0,0,0) but object_id=%s exists, using reach_for with object_id only",
                        target_id,
                    )
                # Drop the bogus position so the id-only path is taken.
                target_pos = None
                meta["navigation"] = "reach_for_with_object_id"
                meta["note"] = "invalid_position_using_object_id_reach_for"
            else:
                # Without an id and without a task target there is nothing to act on.
                if self.logger:
                    self.logger.warning(
                        "[Executor] rejecting pick with invalid target_pos (0,0,0) and no valid object_id: target_id=%s",
                        target_id,
                    )
                meta.update({"reason": "invalid_target_position"})
                return {"type": "ongoing"}, meta
        
        # Start navigating before checking the id: the depth cutoff keeps distant
        # objects out of object_info, but they are still worth approaching.
        navigation_started = False
        navigation_command = None
        
        # Navigate if target_pos is available
        if target_pos is not None:
            if is_force_pick and force_pick_distance is not None and force_pick_distance <= 2.0:
                if self.logger:
                    self.logger.info(
                        "[Executor] force_pick: skipping navigation (dist=%.2f <= 2.0m), attempting direct pick",
                        force_pick_distance,
                    )
                meta["navigation"] = "skip_direct_pick"
                meta["distance"] = force_pick_distance
            elif not is_force_pick:
                # Use smaller arrived_at (0.3m) for pick actions to get closer to objects
                navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True, arrived_at=0.3)
                meta.update(nav_meta)
                navigation_started = True
                if nav_meta.get("forced_failure"):
                    if self.logger and getattr(self.logger, "debug", None):
                        self.logger.debug(
                            "[Executor] pick aborted due to guard meta=%s",
                            nav_meta,
                        )
                    return navigation_command, meta
            else:  # is_force_pick and distance > 2.0
                # Use smaller arrived_at (0.3m) for pick actions to get closer to objects
                navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True, arrived_at=0.3)
                meta.update(nav_meta)
                navigation_started = True
                if nav_meta.get("forced_failure"):
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: guard triggered but proceeding anyway (dist=%.2f)",
                            nav_meta.get("distance", -1),
                        )
                    meta["guard_ignored"] = True
        else:
            # Position estimation failed but depth still gives a range, which is
            # enough to start closing the distance.
            is_task_target = plan.meta.get("is_task_target", False)
            retry_navigation = plan.meta.get("reason") == "retry_navigation_after_failure"
            use_reach_for_flag = plan.meta.get("use_reach_for", False)
            
            # An explicit reach_for request skips navigation entirely.
            if use_reach_for_flag:
                if self.logger:
                    self.logger.info(
                        "[Executor] pick: use_reach_for flag set, using reach_for with object_id=%s (no position)",
                        target_id,
                    )
                meta["navigation"] = "reach_for_with_object_id"
                meta["distance"] = force_pick_distance
                meta["note"] = "use_reach_for_flag_set"
                # Will use reach_for (type 3) below - skip to pick section
            elif is_force_pick and force_pick_distance is not None:
                if force_pick_distance <= 2.5:
                    # Close enough for direct pick
                    if self.logger:
                        self.logger.info(
                            "[Executor] force_pick: no position but distance=%.2f <= 2.5m, attempting direct pick",
                            force_pick_distance,
                        )
                    meta["navigation"] = "skip_no_position"
                    meta["distance"] = force_pick_distance
                    meta["note"] = "pick_without_position"
                elif is_task_target or retry_navigation:
                    # Task target or navigation retry - use reach_for (type 3) which automatically navigates to object_id
                    # reach_for will handle navigation automatically when given object_id
                    if self.logger:
                        self.logger.info(
                            "[Executor] pick: no position but task target/retry distance=%.2f, using reach_for with object_id only",
                            force_pick_distance,
                        )
                    meta["navigation"] = "reach_for_with_object_id"
                    meta["distance"] = force_pick_distance
                    meta["note"] = "no_position_using_object_id_reach_for"
                    # Will use reach_for (type 3) below - skip to pick section
                else:
                    # Not task target and too far
                    if self.logger:
                        self.logger.debug(
                            "[Executor] pick: target_pos is None and distance=%.2f > 2.5m (not task target), skipping pick",
                            force_pick_distance,
                        )
                    meta.update({"reason": "no_position_and_too_far"})
                    return {"type": "ongoing"}, meta
            elif retry_navigation and force_pick_distance is not None:
                # Navigation retry: use reach_for even if not force_pick
                if self.logger:
                    self.logger.info(
                        "[Executor] pick: navigation retry, no position but distance=%.2f, using reach_for with object_id only",
                        force_pick_distance,
                    )
                meta["navigation"] = "reach_for_with_object_id"
                meta["distance"] = force_pick_distance
                meta["note"] = "no_position_using_object_id_reach_for"
                # Will use reach_for (type 3) below - skip to pick section
            else:
                # Position is None and distance is unknown
                # But if it's a task target and object_id exists, try reach_for anyway
                if is_task_target and plan.target_id is not None:
                    if self.logger:
                        self.logger.info(
                            "[Executor] pick: target_pos is None and distance unknown, but is_task_target=True and object_id=%s exists, using reach_for",
                            plan.target_id,
                        )
                    meta["navigation"] = "reach_for_with_object_id"
                    meta["note"] = "no_position_using_object_id_reach_for"
                    # Will use reach_for (type 3) below - skip to pick section
                else:
                    if self.logger:
                        self.logger.debug(
                            "[Executor] pick: target_pos is None and distance unknown, skipping pick",
                        )
                    meta.update({"reason": "no_position_and_no_distance"})
                    return {"type": "ongoing"}, meta
        
        # Now attempt pick
        target_id = plan.target_id
        if target_id is None:
            meta.update({"reason": "missing_target_id"})
            return {"type": "ongoing"}, meta
        
        # Convert to int and check if it's valid (0 is not a valid object_id)
        try:
            target_id_int = int(target_id)
            if target_id_int == 0:
                if self.logger:
                    self.logger.warning(
                        "[Executor] target_id=%s is 0 (invalid), skipping pick (may cause KeyError)",
                        target_id,
                    )
                meta.update({"reason": "invalid_target_id_zero"})
                return {"type": "ongoing"}, meta
        except (ValueError, TypeError) as exc:
            if self.logger:
                self.logger.warning(
                    "[Executor] target_id=%s cannot be converted to int: %s",
                    target_id,
                    exc,
                )
            meta.update({"reason": "invalid_target_id_type"})
            return {"type": "ongoing"}, meta
        
        # Once navigation is under way, let it finish and retry the pick later
        # rather than aborting because the object is not in object_info yet.
        if self.agent_memory is not None and hasattr(self.agent_memory, "object_info"):
            try:
                if target_id_int not in self.agent_memory.object_info:
                    is_task_target = plan.meta.get("is_task_target", False)
                    use_reach_for_flag = plan.meta.get("use_reach_for", False)
                    # reach_for works from the object id alone, so an object missing
                    # from memory is not a reason to reject it.
                    if use_reach_for_flag or (target_pos is None and meta.get("note") in ("no_position_using_object_id_reach_for", "use_reach_for_flag_set")):
                        if self.logger:
                            self.logger.info(
                                "[Executor] object_id=%s not in agent_memory.object_info but will use reach_for (type 3), skipping object_info check (use_reach_for=%s, note=%s)",
                                target_id_int,
                                use_reach_for_flag,
                                meta.get("note"),
                            )
                    # Navigation already in flight: keep going.
                    elif navigation_started and navigation_command is not None:
                        if self.logger:
                            self.logger.info(
                                "[Executor] object_id=%s not in agent_memory.object_info but navigation already started, continuing navigation (task_target=%s)",
                                target_id_int,
                                is_task_target,
                            )
                        meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                        return navigation_command, meta
                    # A task target with a position is always worth approaching.
                    elif is_task_target and target_pos is not None:
                        if self.logger:
                            self.logger.info(
                                "[Executor] object_id=%s not in agent_memory.object_info but is task target, allowing navigation",
                                target_id_int,
                            )
                        # Navigation is in flight or about to start.
                        if navigation_command is not None:
                            meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                            return navigation_command, meta
                        # Nothing in flight yet: start navigating.
                        else:
                            navigation_command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                            meta.update(nav_meta)
                            meta.update({"reason": "object_not_in_memory_but_navigating", "will_retry_pick_after_nav": True})
                            return navigation_command, meta
                    else:
                        # Neither navigating nor a task target: skip it.
                        if self.logger:
                            self.logger.warning(
                                "[Executor] object_id=%s not in agent_memory.object_info, skipping pick (may cause KeyError)",
                                target_id_int,
                            )
                        meta.update({"reason": "object_not_in_memory"})
                        return {"type": "ongoing"}, meta
            except Exception as exc:
                if self.logger:
                    self.logger.debug(
                        "[Executor] failed to check agent_memory.object_info for id=%s: %s",
                        target_id_int,
                        exc,
                    )
        
        # Check actual distance before issuing pick command
        # TDW reach_threshold is 2m, so we need to be within 2m to successfully pick
        actual_distance = float("inf")
        if self.agent_memory is not None and hasattr(self.agent_memory, "position") and self.agent_memory.position is not None and target_pos is not None:
            try:
                agent_pos = self.agent_memory.position
                if isinstance(agent_pos, (list, tuple, np.ndarray)) and len(agent_pos) >= 3:
                    agent_arr = np.array([agent_pos[0], agent_pos[2]])
                    target_arr = np.array([target_pos[0], target_pos[2] if len(target_pos) > 2 else target_pos[1]])
                    actual_distance = float(np.linalg.norm(agent_arr - target_arr))
            except Exception as exc:
                if self.logger:
                    self.logger.debug("[Executor] pick: failed to calculate actual distance: %s", exc)
        
        # Use plan.meta distance if actual_distance calculation failed
        if actual_distance == float("inf") and plan.meta.get("distance") is not None:
            actual_distance = float(plan.meta.get("distance"))
        
        reach_threshold = 2.5
        
        # reach_for navigates on its own, so with only an object id the distance
        # check is skipped and the command is issued directly.
        use_reach_for_flag = plan.meta.get("use_reach_for", False)
        reach_for_notes = ("no_position_using_object_id_reach_for", "invalid_position_using_object_id_reach_for", "use_reach_for_flag_set")
        if use_reach_for_flag or (target_pos is None and meta.get("note") in reach_for_notes):
            # Use reach_for (type 3) which will automatically navigate to object_id
            holding_slots = agent_state.get("holding_slots", {}) if isinstance(agent_state, dict) else {}
            left_busy = holding_slots.get("left") is not None
            right_busy = holding_slots.get("right") is not None
            if left_busy and right_busy:
                meta.update({"reason": "hands_full"})
                return {"type": "ongoing"}, meta
            arm = "right" if left_busy and not right_busy else "left"
            command = {"type": 3, "object": target_id_int, "arm": arm}
            meta.update({"result": "attempt_pick_with_reach_for", "arm": arm, "distance": force_pick_distance})
            if self.logger:
                # Handle None distance for logging
                dist_str = f"{force_pick_distance:.2f}" if force_pick_distance is not None else "unknown"
                self.logger.info(
                    "[Executor] using reach_for (type 3) with object_id only: id=%s arm=%s dist=%s",
                    target_id_int,
                    arm,
                    dist_str,
                )
            return command, meta
        
        if actual_distance > reach_threshold:
            if self.logger:
                self.logger.debug(
                    "[Executor] pick: too far from target (dist=%.2f > %.2f), continuing navigation instead of pick",
                    actual_distance,
                    reach_threshold,
                )
            # Continue navigation instead of attempting pick
            if target_pos is not None:
                command, nav_meta = self._navigate_to(target_pos, agent_state, follow=True, skip_clamp=True)
                meta.update(nav_meta)
                meta.update({"reason": "too_far_for_pick", "distance": actual_distance, "threshold": reach_threshold})
                return command, meta
            else:
                meta.update({"reason": "too_far_for_pick_no_position", "distance": actual_distance, "threshold": reach_threshold})
                return {"type": "ongoing"}, meta
        
        holding_slots = agent_state.get("holding_slots", {}) if isinstance(agent_state, dict) else {}
        left_busy = holding_slots.get("left") is not None
        right_busy = holding_slots.get("right") is not None
        if left_busy and right_busy:
            meta.update({"reason": "hands_full"})
            return {"type": "ongoing"}, meta
        arm = "right" if left_busy and not right_busy else "left"
        command = {"type": 3, "object": target_id_int, "arm": arm}
        meta.update({"result": "attempt_pick", "arm": arm, "distance": actual_distance})
        if self.logger and getattr(self.logger, "debug", None):
            self.logger.debug("[Executor] issuing pick command=%s meta=%s (dist=%.2f <= %.2f)", command, meta, actual_distance, reach_threshold)
        return command, meta

    def _guard_threshold(self) -> Optional[float]:
        if self.agent_memory is None or not hasattr(self.agent_memory, "map_size"):
            return None
        map_size = getattr(self.agent_memory, "map_size")
        if not map_size:
            return None
        return float(min(map_size)) * self.cfg.navigation_guard_ratio

    def tick(self):
        removal = []
        for key, value in self._guard_skip.items():
            value["frames"] -= 1
            if value["frames"] <= 0:
                removal.append(key)
        for key in removal:
            self._guard_skip.pop(key, None)

    def guard_summary(self):
        return {key: value["frames"] for key, value in self._guard_skip.items()}

    # -- target validation ---------------------------------------------------
    def _agent_position(self):
        """Current agent position as ((x, z), (x, y, z)), or (None, None)."""
        position = getattr(self.agent_memory, "position", None)
        if position is None:
            return None, None
        try:
            if len(position) < 3:
                return None, None
            return (float(position[0]), float(position[2])), position
        except (TypeError, ValueError):
            return None, None

    def _scene_bounds(self):
        bounds = getattr(self.agent_memory, "_scene_bounds", None)
        if not bounds:
            return None
        values = [bounds.get(key) for key in ("x_min", "x_max", "z_min", "z_max")]
        return None if any(value is None for value in values) else values

    def _clamp_to_bounds(self, coord, margin=0.0):
        bounds = self._scene_bounds()
        if bounds is None:
            return coord
        x_min, x_max, z_min, z_max = bounds
        x = min(max(coord[0], x_min + margin), x_max - margin)
        z = min(max(coord[1], z_min + margin), z_max - margin)
        return (x, z)

    def _in_room(self, coord) -> bool:
        """Whether the environment considers this (x, z) inside the floorplan.

        A missing API means we cannot tell, and the position is accepted.
        """
        if not self.env_api or "check_pos_in_room" not in self.env_api:
            return True
        try:
            return bool(self.env_api["check_pos_in_room"]((coord[0], coord[1])))
        except Exception:
            return False

    def _scene_centre(self):
        bounds = self._scene_bounds()
        if bounds is None:
            return None
        x_min, x_max, z_min, z_max = bounds
        centre = ((x_min + x_max) / 2.0, (z_min + z_max) / 2.0)
        return centre if self._in_room(centre) else None

    def _room_centre(self, agent_xyz, agent_xz):
        """Centre of the room the agent is standing in, if it can be resolved."""
        if agent_xz is None:
            return None
        if not hasattr(self.agent_memory, "belongs_to_which_room"):
            return None
        if not hasattr(self.agent_memory, "center_of_room"):
            return None
        try:
            reference = agent_xyz if agent_xyz is not None else (agent_xz[0], 0, agent_xz[1])
            room = self.agent_memory.belongs_to_which_room(reference)
            if room is None:
                return None
            centre = self.agent_memory.center_of_room(room)
        except Exception:
            return None
        if centre is None:
            return None
        candidate = (centre[0], centre[2])
        return candidate if self._in_room(candidate) else None

    def _search_near(self, agent_xz, max_radius=5.0, step=1.0):
        """First in-bounds, in-room position found on rings around the agent."""
        if agent_xz is None:
            return None
        bounds = self._scene_bounds()
        radius = step
        while radius <= max_radius:
            for angle in range(0, 360, 45):
                radians = math.radians(angle)
                candidate = (
                    agent_xz[0] + radius * math.cos(radians),
                    agent_xz[1] + radius * math.sin(radians),
                )
                if bounds is not None:
                    x_min, x_max, z_min, z_max = bounds
                    if not (x_min <= candidate[0] <= x_max and z_min <= candidate[1] <= z_max):
                        continue
                if self._in_room(candidate):
                    return candidate
            radius += step
        return None

    def _clamp_target(self, target_pos):
        """Move a target inside the scene bounds and into a reachable room.

        Anything outside the floorplan is replaced by the nearest sensible
        alternative: the agent itself, the scene centre, the centre of the room
        the agent is in, or a point found by searching rings around the agent.
        """
        if self.agent_memory is None:
            return target_pos
        x, y, z = target_pos
        agent_xz, agent_xyz = self._agent_position()

        clamped = self._clamp_to_bounds((x, z), margin=0.5)
        if not self._in_room(clamped):
            clamped = agent_xz or (0.0, 0.0)
            if not self._in_room(clamped):
                fallback = (
                    self._scene_centre()
                    or self._room_centre(agent_xyz, agent_xz)
                    or self._search_near(agent_xz)
                )
                if fallback is not None:
                    clamped = fallback
                elif self.logger:
                    self.logger.warning(
                        "[Executor] target %s outside the floorplan and no valid fallback found",
                        target_pos,
                    )

        clamped = self._clamp_to_bounds(clamped)
        if not self._in_room(clamped):
            fallback = self._scene_centre() or self._room_centre(agent_xyz, agent_xz)
            if fallback is not None:
                clamped = fallback
            elif self.logger:
                self.logger.warning(
                    "[Executor] clamped position %s is outside the floorplan (target %s)",
                    clamped,
                    target_pos,
                )

        if (clamped[0], clamped[1]) != (x, z) and self.logger:
            self.logger.debug("[Executor] clamp target %s -> %s", target_pos, clamped)
        return (clamped[0], y, clamped[1])
