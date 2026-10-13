"""Grounding tests that do not need CLIP weights or the TDW build.

The CLIP encoder is replaced by a deterministic stub so the tests exercise the
region proposal and the top-K selection rather than the pretrained model.
"""

from __future__ import annotations

import types
import unittest

import numpy as np
import torch

import vico_lite.perception.clip_encoder as clip_encoder
from vico_lite.perception.regions import extract_object_patches

DIM = 512


class _StubClipModel:
    def requires_grad_(self, flag):
        return self

    def encode_image(self, batch):
        features = torch.zeros(batch.shape[0], DIM)
        for index in range(batch.shape[0]):
            features[index, 0] = float(batch[index].mean())
            features[index, 1] = 1.0
        return features

    def encode_text(self, tokens):
        return torch.ones(tokens.shape[0], DIM)


def _stub_preprocess(image):
    return torch.full((3, 4, 4), float(np.asarray(image).mean()) / 255.0)


def _install_stub():
    clip_encoder.load_clip = lambda name, device="cpu": (_StubClipModel(), _stub_preprocess)
    clip_encoder.clip = types.SimpleNamespace(
        tokenize=lambda texts, truncate=False: torch.zeros(len(texts), 7, dtype=torch.long)
    )


class RegionProposalTest(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        self.seg = np.zeros((64, 64, 3), dtype=np.int32)
        self.seg[10:20, 30:44] = (255, 0, 0)
        self.seg[50:52, 5:7] = (0, 255, 0)
        self.seg[0:1, 0:1] = (0, 0, 255)
        self.infos = [
            {"name": "apple", "seg_color": (255, 0, 0)},
            {"name": "bread", "seg_color": (0, 255, 0)},
            {"name": "speck", "seg_color": (0, 0, 255)},
            {"name": "unsegmented", "seg_color": None},
        ]

    def test_drops_objects_with_too_few_pixels(self):
        _, indices = extract_object_patches(self.rgb, self.seg, self.infos)
        self.assertEqual(indices, [0, 1])

    def test_patches_are_large_enough_to_encode(self):
        patches, _ = extract_object_patches(self.rgb, self.seg, self.infos)
        for patch in patches:
            self.assertGreaterEqual(min(patch.shape[:2]), 16)

    def test_no_segmentation_yields_no_patches(self):
        self.assertEqual(extract_object_patches(self.rgb, None, self.infos), ([], []))


class PipelineTest(unittest.TestCase):
    def setUp(self):
        _install_stub()
        from vico_lite.config import ViCoConfig
        from vico_lite.perception.pipeline import PerceptionAlignmentPipeline

        self.cfg = ViCoConfig()
        self.pipeline = PerceptionAlignmentPipeline(self.cfg)

        # The environment hands out RGB as 3xHxW.
        self.rgb = np.zeros((3, 48, 48), dtype=np.uint8)
        self.rgb[:, 10:20, 10:30] = 200
        self.rgb[:, 30:40, 5:15] = 60
        self.seg = np.zeros((48, 48, 3), dtype=np.int32)
        self.seg[10:20, 10:30] = (1, 0, 0)
        self.seg[30:40, 5:15] = (0, 1, 0)
        self.infos = [
            {"id": 1, "name": "bright", "seg_color": (1, 0, 0)},
            {"id": 2, "name": "dim", "seg_color": (0, 1, 0)},
        ]

    def _process(self):
        return self.pipeline.process(
            {"rgb": self.rgb, "seg_mask": self.seg, "object_infos": self.infos},
            "Collect 2 x bright",
        )

    def test_each_object_is_scored_separately(self):
        candidates = self._process().symbolic[0]["candidates"]
        self.assertEqual([c["name"] for c in candidates], ["bright", "dim"])
        self.assertNotEqual(candidates[0]["score"], candidates[1]["score"])

    def test_top_k_bounds_the_candidate_count(self):
        self.cfg.symbolic_top_k = 1
        self.assertEqual(len(self._process().symbolic[0]["candidates"]), 1)

    def test_blind_frame_is_handled(self):
        output = self.pipeline.process(
            {"rgb": self.rgb, "seg_mask": None, "object_infos": []}, "x"
        )
        self.assertEqual(output.symbolic[0]["candidates"], [])

    def test_kosmos_is_off_by_default(self):
        self.assertIsNone(self.pipeline.kosmos)
        self.assertNotIn("kosmos_feature", self._process().symbolic[0]["candidates"][0])


if __name__ == "__main__":
    unittest.main()
