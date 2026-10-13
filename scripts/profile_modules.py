"""Per-module CPU latency of the ViCo-Lite pipeline (paper Table IV).

Each module is timed in isolation on synthetic inputs shaped like a TDW-MAT
step: a 512x512 RGB frame with ``--objects`` segmented objects, two agents
sharing one TeamMemoryHub, and A* on a 240x240 cost grid. Perception uses the
real frozen CLIP ViT-B/32, so the weights must be available.

Table V is derived from these means: per-episode latency is the mean latency
times the calls per step times the average episode length.

    python scripts/profile_modules.py --repeats 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vico_lite.config import ViCoConfig  # noqa: E402
from vico_lite.memory import TeamMemoryHub  # noqa: E402
from vico_lite.perception.pipeline import PerceptionAlignmentPipeline  # noqa: E402
from vico_lite.scoring import candidate_score  # noqa: E402

FRAME = 512
GRID = 240


def synthetic_observation(rng: np.random.Generator, n_objects: int) -> Dict:
    rgb = rng.integers(0, 256, size=(3, FRAME, FRAME), dtype=np.uint8)
    seg = np.zeros((FRAME, FRAME, 3), dtype=np.int32)
    infos = []
    for index in range(n_objects):
        top, left = rng.integers(0, FRAME - 64, size=2)
        colour = (index + 1, 0, 0)
        seg[top:top + 48, left:left + 48] = colour
        infos.append({
            "id": index,
            "name": f"object_{index}",
            "seg_color": colour,
            "position": [float(rng.uniform(-10, 10)), 0.0, float(rng.uniform(-5, 5))],
            "distance": float(rng.uniform(0.5, 10.0)),
        })
    return {"rgb": rgb, "seg_mask": seg, "object_infos": infos, "instruction": "Collect 2 x apple, 1 x bread"}


def mean_ms(fn: Callable[[], object], repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--objects", type=int, default=6, help="visible objects per frame")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    rng = np.random.default_rng(0)
    cfg = ViCoConfig()
    pipeline = PerceptionAlignmentPipeline(cfg, device="cpu")
    observation = synthetic_observation(rng, args.objects)
    perception = pipeline.process(observation, observation["instruction"])

    hub = TeamMemoryHub(cfg)
    hub.begin_episode(0)
    hub.update_skip_targets(0, [], [(1.0, 1.0)], [3])
    hub.update_obstacles(1, [(2.0, 2.0)], current_frame=0)
    frame = iter(range(10**9))

    def shared_memory_update():
        hub._frame = 0
        hub._synced_frame = 0  # Isolate the per-agent object merge.
        hub._merge_objects(0, perception.symbolic)

    def team_hub_update():
        hub._frame = next(frame)
        hub._sync_team()

    candidates = perception.symbolic[0]["candidates"]
    weights = cfg.reasoner_heuristic_weights

    def heuristic_scoring():
        return max(candidates, key=lambda c: candidate_score(c["distance"], c["score"], weights))

    import pyastar2d

    cost = np.ones((GRID, GRID), dtype=np.float32)
    cost[rng.random((GRID, GRID)) < 0.15] = 1000.0

    def astar():
        return pyastar2d.astar_path(cost, (5, 5), (GRID - 6, GRID - 6), allow_diagonal=False)

    rows = [
        ("Full perception (CLIP + symbolic candidates)",
         mean_ms(lambda: pipeline.process(observation, observation["instruction"]), max(args.repeats // 10, 10))),
        ("Shared-memory update", mean_ms(shared_memory_update, args.repeats)),
        ("Team hub update", mean_ms(team_hub_update, args.repeats)),
        ("Heuristic scoring", mean_ms(heuristic_scoring, args.repeats)),
        (f"A* pathfinding ({GRID} x {GRID} grid)", mean_ms(astar, args.repeats)),
    ]
    width = max(len(name) for name, _ in rows)
    print(f"{'Module':<{width}}  Mean latency (ms)")
    for name, value in rows:
        print(f"{name:<{width}}  {value:.3f}")


if __name__ == "__main__":
    main()
