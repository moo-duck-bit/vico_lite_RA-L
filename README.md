# ViCo-Lite


<img width="3168" height="1344" alt="ViCo-Lite architecture" src="https://github.com/user-attachments/assets/7dad0bb4-2525-4f8c-b8f2-e104e382bd81" />

# ViCo-Lite

**Accepted to IEEE Robotics and Automation Letters (RA-L) SCI Q1 — 11 September 2026.**

Training-free multi-agent AI agents for embodied cooperation. Two agents explore a
multi-room house, find the objects a natural-language instruction names, and carry
them to a goal — with no task-specific training and **zero LLM calls during
execution**.

<img width="3168" height="1344" alt="ViCo-Lite architecture" src="https://github.com/user-attachments/assets/7dad0bb4-2525-4f8c-b8f2-e104e382bd81" />

| | |
| --- | --- |
| Benchmark | ThreeDWorld Multi-Agent Transport (TDW-MAT), 24 episodes |
| Transport rate | **88** with oracle perception, **84** without (CoELA: 83 / 70) |
| LLM calls at run time | **0** |
| Training | none — frozen CLIP ViT-B/32, optional Kosmos-2 |
| Hardware | CPU only, no GPU and no external API required |

## Design

The system is built around three decisions, each of which is measured rather than
assumed.

**1. Put the complexity in perception, not in planning.** Comparable systems
(CoELA, CaPo, CoTS) convert a scene into symbolic text — `"apple"`, `"on table"` —
and then compensate for the lost detail with multi-round LLM deliberation. ViCo-Lite
inverts this: each agent crops one image patch per detected object, scores it
against the instruction with a frozen CLIP encoder, and lets a deterministic policy
act on the result. Grounding improves and the planner gets simpler at the same time.

**2. Treat the LLM as an interface, not as the control loop.** An LLM planner in the
hot path costs an API round trip on every decision, across thousands of steps per
episode. Here the LLM branch is a real, working component — prompt template, JSON
schema, candidate re-ranking, frame-based cooldown — but it is **off by default**,
and the deterministic path carries every reported result. Per-decision latency
becomes bounded and predictable, inference cost goes to zero, and no observation
ever leaves the machine.

**3. Coordinate through shared state, not through conversation.** Agents publish
into a shared object table: label, 3D position, relevance score, last-observed step,
task status. Positions and scores from different agents are blended rather than
overwritten. The update is O(1) in the number of agents, so coordination cost does
not grow as the team does.

The ablations say this is the right ordering. Removing shared memory drops completion
from 92% to 30%; removing the ranking heuristic changes nothing. Coordination is what
carries the system, and it is the cheap part.

## Efficiency

Per-module latency, measured on CPU:

| Module | Mean latency |
| --- | --- |
| Perception (CLIP + candidate construction) | 72.61 ms |
| A\* pathfinding | 4.91 ms |
| Candidate scoring | 0.066 ms |
| Shared-memory update | 0.037 ms |
| Team latent EMA | 0.012 ms |

Over an average 620-step episode this is ≈ 91 s of compute, of which perception is
over 99% and **all coordination and decision-making together is under 0.15%**. An
LLM-in-the-loop planner would add seconds per decision on top.

## Architecture

Each package maps to one component of the published method.

| Component | Module |
| --- | --- |
| Perception — region proposal, CLIP relevance scoring | [`vico_lite/perception/`](vico_lite/perception/) |
| Visual-language grounding — frozen encoders, no trained head | [`vico_lite/perception/pipeline.py`](vico_lite/perception/pipeline.py) |
| Shared memory — object table, team latent EMA | [`vico_lite/memory/hub.py`](vico_lite/memory/hub.py) |
| Language — optional LLM proposal branch | [`vico_lite/language/`](vico_lite/language/) |
| Policy — candidate generation and ranking | [`vico_lite/policy/policy.py`](vico_lite/policy/policy.py) |
| Execution — simulator commands, A\*, failure feedback | [`vico_lite/execution/executor.py`](vico_lite/execution/executor.py) |

Object relevance is the cosine similarity between a CLIP image-patch embedding and
the instruction embedding. Candidates are then ranked by

```
s(a) = w_d / (d + eps) + (w_n + w_v) * c       w_d=1.5, w_n=0.8, w_v=0.6
```

implemented once and shared by both ranking paths. The weights come from the task
structure and are not tuned; `scripts/ablations/score_weights.sh` sweeps them.

## Results

Transport rate on the 24-episode TDW-MAT test set. Baselines are GPT-4 driven and
query the model throughout the episode; ViCo-Lite queries nothing.

| | Food | Stuff | Average |
| --- | --- | --- | --- |
| **Oracle perception** | | | |
| CoELA (GPT-4) | 85 | 81 | 83 |
| CoTS (GPT-4) | **94** | **90** | **92** |
| ViCo-Lite | 90 | 87 | 88 |
| **Non-oracle perception** | | | |
| CoELA (GPT-4) | 79 | 60 | 70 |
| CoTS (GPT-4) | **89** | 76 | 83 |
| ViCo-Lite | 85 | **83** | **84** |

The gap between the two settings is what matters: CoTS loses 9 points when
perception degrades, ViCo-Lite loses 4. Grounding in a continuous visual-language
space degrades more gracefully than reasoning over symbolic text.

## Repository layout

```
vico_lite/     agent stack — perception, memory, language, policy, execution
tdw-gym/       simulator wrapper, evaluation entry point, baseline agents
scripts/       reproduction scripts for every table in the paper
tests/         unit tests that need neither model weights nor the simulator
docs/          benchmark reference and low-level controller API
dataset/       scene and task configurations
```

## Setup

```bash
conda create -n vico_lite python=3.9 && conda activate vico_lite
pip install -e .
pip install git+https://github.com/openai/CLIP.git
```

On a headless server, follow the
[TDW installation guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md)
to configure the X server, then verify with `python demo/demo_scene.py`.

Non-oracle perception additionally needs `mmdetection` and the fine-tuned
segmentation weights; see [`docs/benchmark.md`](docs/benchmark.md).

## Reproducing the results

```bash
bash scripts/eval_oracle.sh              # 24 episodes, ground-truth segmentation
bash scripts/eval_no_oracle.sh           # 24 episodes, segmentation estimated from RGB
bash scripts/ablations/shared_memory.sh  # shared memory / ranking, on and off
bash scripts/ablations/top_k.sh          # retained candidate count
bash scripts/ablations/perception.sh     # graded perception strength, down to blind
bash scripts/ablations/score_weights.sh  # relevance weights of s(a)
```

Every configuration is a flag on `tdw-gym/challenge.py`, so a result and the command
that produced it stay in one place:

| Flag | Effect |
| --- | --- |
| `--no_gt_mask` | Estimate labels and masks from RGB |
| `--seg_dropout P` | Drop a fraction `P` of detected objects per frame |
| `--symbolic_top_k K` | Candidates retained per frame |
| `--no_shared_memory` | Give each agent its own memory |
| `--no_heuristic_scoring` | Rank by distance alone |
| `--enable_llm_reasoner` | Turn on the LLM branch (off in all reported results) |

## Tests

```bash
python -m unittest discover -s tests
```

The suite stubs the CLIP encoder, so it runs in under a second without model
weights, without a GPU and without the simulator — covering region proposal, top-K
selection, the shared object table and the candidate score.

## Acknowledgements

Built on [CoELA / Co-LLM-Agents](https://github.com/UMass-Foundation-Model/Co-LLM-Agents)
(TDW-MAT environment and baselines) and
[transport_challenge_multi_agent](https://github.com/alters-mit/transport_challenge_multi_agent)
by Esther Alter, vendored under its MIT licence.
