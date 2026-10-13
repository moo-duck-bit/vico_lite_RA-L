"""Shared-memory tests: object table, task status and per-step team sync."""

from __future__ import annotations

import unittest

from vico_lite.config import ViCoConfig
from vico_lite.memory import TeamMemoryHub
from vico_lite.perception.pipeline import PerceptionOutput


def observation(candidates) -> PerceptionOutput:
    return PerceptionOutput(symbolic=[{"candidates": candidates}])


class TeamMemoryHubTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ViCoConfig()
        self.hub = TeamMemoryHub(self.cfg)
        self.hub.begin_episode(0)
        self.apple = {
            "id": 7,
            "name": "apple",
            "position": [1.0, 0.0, 1.0],
            "score": 0.8,
            "is_task_target": True,
        }

    def test_entries_store_label_position_score_time_and_status(self):
        entry = self.hub.update(0, observation([self.apple]), frame=42).team_symbolic[0]
        self.assertEqual(entry["name"], "apple")
        self.assertEqual(entry["position"], [1.0, 0.0, 1.0])
        self.assertEqual(entry["score"], 0.8)
        self.assertEqual(entry["last_seen_step"], 42, "tau_j is the environment step")
        self.assertEqual(entry["status"], "pending")

    def test_the_same_object_seen_twice_is_smoothed_not_duplicated(self):
        self.hub.update(0, observation([self.apple]), frame=1)
        second = dict(self.apple, position=[3.0, 0.0, 3.0], score=0.4)
        del second["is_task_target"]
        snapshot = self.hub.update(1, observation([second]), frame=1)

        self.assertEqual(len(snapshot.team_symbolic), 1)
        entry = snapshot.team_symbolic[0]
        self.assertEqual(entry["position"], [2.0, 0.0, 2.0])
        self.assertAlmostEqual(entry["score"], 0.6)
        self.assertEqual(entry["observed_by"], [0, 1])
        self.assertTrue(entry["is_task_target"], "flags from the first observer must survive")

    def test_objects_without_id_merge_within_a_half_metre_cell(self):
        mug = {"name": "mug", "position": [2.1, 0.0, 4.1], "score": 0.5}
        self.hub.update(0, observation([mug]), frame=1)
        snapshot = self.hub.update(1, observation([dict(mug, position=[1.9, 0.0, 3.9])]), frame=1)
        self.assertEqual(len(snapshot.team_symbolic), 1)

    def test_delivered_objects_are_done_for_every_agent_from_the_next_step(self):
        self.hub.update(0, observation([self.apple]), frame=1)
        self.hub.update_skip_targets(0, [], [], [7])
        same_step = self.hub.update(1, observation([]), frame=1)
        self.assertEqual(same_step.team_symbolic[0]["status"], "pending")
        next_step = self.hub.update(1, observation([]), frame=2)
        self.assertEqual(next_step.team_symbolic[0]["status"], "done")
        self.assertEqual(next_step.skip_targets["ids"], [7])

    def test_status_is_tracked_by_id_not_by_name(self):
        second_apple = dict(self.apple, id=8, position=[5.0, 0.0, 5.0])
        self.hub.update(0, observation([self.apple, second_apple]), frame=1)
        self.hub.update_skip_targets(0, ["apple"], [], [7])
        status = {e["id"]: e["status"] for e in self.hub.update(0, observation([]), frame=2).team_symbolic}
        self.assertEqual(status, {7: "done", 8: "pending"})

    def test_team_sync_runs_once_per_environment_step(self):
        calls = []
        original = self.hub._sync_team
        self.hub._sync_team = lambda: (calls.append(1), original())
        for frame in (1, 2, 3):
            self.hub.update(0, observation([]), frame=frame)
            self.hub.update(1, observation([]), frame=frame)
        self.assertEqual(len(calls), 3)

    def test_obstacles_expire_after_their_ttl(self):
        self.hub.update_obstacles(0, [(1.0, 1.0)], current_frame=0)
        self.assertEqual(len(self.hub.update(0, observation([]), frame=119).obstacle_coords), 1)
        self.assertEqual(len(self.hub.update(0, observation([]), frame=120).obstacle_coords), 0)

    def test_a_new_episode_starts_from_an_empty_table(self):
        self.hub.update(0, observation([self.apple]), frame=1)
        self.hub.begin_episode(1)
        self.assertEqual(self.hub.update(0, observation([]), frame=0).team_symbolic, [])

    def test_coordinates_are_quantised_to_half_a_metre(self):
        self.hub.update_skip_targets(0, ["apple"], [(1.13, -2.31)])
        snapshot = self.hub.update(0, observation([]), frame=1)
        self.assertEqual(snapshot.skip_targets["coords"], [[1.0, -2.5]])


class ConfigTest(unittest.TestCase):
    def test_defaults_issue_no_llm_call(self):
        cfg = ViCoConfig()
        self.assertTrue(cfg.force_heuristics)
        self.assertFalse(cfg.use_reasoner)
        self.assertFalse(cfg.use_kosmos)


if __name__ == "__main__":
    unittest.main()
