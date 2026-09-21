"""Candidate scoring: s(a) = w_d / (d + eps) + (w_n + w_v) * c."""

from __future__ import annotations

import unittest

from vico_lite.config import ViCoConfig
from vico_lite.language.reasoner import PolicyReasoner


class CandidateScoreTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ViCoConfig()
        self.reasoner = PolicyReasoner(self.cfg)

    def score(self, distance, relevance):
        return self.reasoner._score_candidate(distance=distance, similarity=relevance)

    def test_matches_the_published_weights(self):
        weights = self.cfg.reasoner_heuristic_weights
        self.assertEqual(weights["distance"], 1.5)
        self.assertEqual(weights["novelty"], 0.8)
        self.assertEqual(weights["visibility"], 0.6)
        expected = 1.5 / (2.0 + 1e-3) + (0.8 + 0.6) * 0.5
        self.assertAlmostEqual(self.score(2.0, 0.5), expected)

    def test_the_nearer_object_always_wins_at_equal_relevance(self):
        self.assertGreater(self.score(1.0, 0.5), self.score(2.0, 0.5))

    def test_relevance_breaks_ties_at_equal_distance(self):
        self.assertGreater(self.score(2.0, 0.9), self.score(2.0, 0.1))

    def test_relevance_only_overturns_a_small_distance_gap(self):
        """Relevance can reorder candidates only while the inverse-distance gap
        stays below w_n + w_v, which is what keeps transport distance-driven."""
        margin = (
            self.cfg.reasoner_heuristic_weights["novelty"]
            + self.cfg.reasoner_heuristic_weights["visibility"]
        )
        gap = 1.5 / 1.0 - 1.5 / 2.0  # 0.75, below the margin
        self.assertLess(gap, margin)
        self.assertLess(self.score(1.0, 0.0), self.score(2.0, 1.0))

        gap = 1.5 / 0.5 - 1.5 / 5.0  # 2.7, above the margin
        self.assertGreater(gap, margin)
        self.assertGreater(self.score(0.5, 0.0), self.score(5.0, 1.0))

    def test_disabling_heuristic_scoring_leaves_pure_distance(self):
        self.cfg.heuristic_scoring = False
        self.assertAlmostEqual(self.score(2.0, 0.9), self.score(2.0, 0.1))
        self.assertAlmostEqual(self.score(2.0, 0.9), 1.5 / (2.0 + 1e-3))


class LlmBranchTest(unittest.TestCase):
    def test_no_client_is_created_under_the_default_config(self):
        self.assertIsNone(PolicyReasoner(ViCoConfig()).client)

    def test_cooldown_is_counted_in_frames(self):
        cfg = ViCoConfig()
        self.assertEqual(PolicyReasoner(cfg).last_call_frame, -cfg.reasoner_min_interval)


if __name__ == "__main__":
    unittest.main()
