"""Shared-memory tests: object table, latent EMA and episode isolation."""

from __future__ import annotations

import unittest

import torch

from vico_lite.config import ViCoConfig
from vico_lite.memory import TeamMemoryHub
from vico_lite.perception.pipeline import PerceptionOutput


def observation(cfg: ViCoConfig, candidates) -> PerceptionOutput:
    return PerceptionOutput(
        vision_latent=torch.zeros(cfg.clip_dim),
        text_latent=torch.zeros(cfg.clip_dim),
        agent_summary=torch.ones(cfg.latent_dim),
        symbolic=[{"candidates": candidates}],
    )


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

    def test_entries_are_flat_objects_the_policy_can_read(self):
        snapshot = self.hub.update(0, observation(self.cfg, [self.apple]))
        entry = snapshot.team_symbolic[0]
        self.assertEqual(entry["name"], "apple")
        self.assertTrue(entry["is_task_target"])
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["last_seen_step"], 1)

    def test_the_same_object_seen_twice_is_merged_not_duplicated(self):
        self.hub.update(0, observation(self.cfg, [self.apple]))
        second = dict(self.apple, position=[3.0, 0.0, 3.0], score=0.4)
        del second["is_task_target"]
        snapshot = self.hub.update(1, observation(self.cfg, [second]))

        self.assertEqual(len(snapshot.team_symbolic), 1)
        entry = snapshot.team_symbolic[0]
        self.assertEqual(entry["position"], [2.0, 0.0, 2.0])
        self.assertAlmostEqual(entry["score"], 0.6)
        self.assertEqual(entry["observed_by"], [0, 1])
        self.assertTrue(entry["is_task_target"], "flags from the first observer must survive")

    def test_skipped_objects_are_marked_done_for_every_agent(self):
        self.hub.update(0, observation(self.cfg, [self.apple]))
        self.hub.update_skip_targets(0, [], [], [7])
        snapshot = self.hub.update(1, observation(self.cfg, []))
        self.assertEqual(snapshot.team_symbolic[0]["status"], "done")
        self.assertEqual(snapshot.skip_targets["ids"], [7])

    def test_team_latent_is_an_ema_of_the_pooled_summaries(self):
        self.hub.update(0, observation(self.cfg, []))
        self.hub.update(1, observation(self.cfg, []))
        self.assertTrue(torch.allclose(self.hub._ema, torch.ones(self.cfg.latent_dim)))

    def test_a_new_episode_starts_from_an_empty_table(self):
        self.hub.update(0, observation(self.cfg, [self.apple]))
        self.hub.begin_episode(1)
        self.assertEqual(self.hub.update(0, observation(self.cfg, [])).team_symbolic, [])

    def test_coordinates_are_quantised_to_half_a_metre(self):
        self.hub.update_skip_targets(0, ["apple"], [(1.13, -2.31)])
        snapshot = self.hub.update(0, observation(self.cfg, []))
        self.assertEqual(snapshot.skip_targets["coords"], [[1.0, -2.5]])


class ConfigTest(unittest.TestCase):
    def test_latent_width_follows_the_encoders(self):
        cfg = ViCoConfig()
        self.assertEqual(cfg.latent_dim, 2 * cfg.clip_dim)
        cfg.kosmos_dim = 64
        self.assertEqual(cfg.latent_dim, 2 * cfg.clip_dim + 64)

    def test_defaults_issue_no_llm_call(self):
        cfg = ViCoConfig()
        self.assertTrue(cfg.force_heuristics)
        self.assertFalse(cfg.use_reasoner)


if __name__ == "__main__":
    unittest.main()
