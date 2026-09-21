# ViCo-Lite

Training-free embodied multi-agent cooperation on ThreeDWorld Multi-Agent
Transport (TDW-MAT). Two agents explore a multi-room house, find the objects a
task names and carry them to a goal position — without any task-specific
training and without a single LLM call during execution.

<img width="3168" height="1344" alt="ViCo-Lite architecture" src="https://github.com/user-attachments/assets/7dad0bb4-2525-4f8c-b8f2-e104e382bd81" />

The design moves complexity out of the planner and into perception. Each agent
crops one image patch per segmented object, scores it against the task
instruction with a frozen CLIP encoder, and writes the result into an
object-level memory that both agents share. Actions are then chosen by a
deterministic scorer over that memory. Nothing is trained, and no autoregressive
model runs inside the control loop.

## Method to code

| Paper | Where it lives |
| --- | --- |
| Perception module — object patches, CLIP scores `c = û·v̂` | [regions.py](vico_lite/perception/regions.py), [clip_encoder.py](vico_lite/perception/clip_encoder.py), [pipeline.py](vico_lite/perception/pipeline.py) |
| Visual-language grounding — frozen encoders, no trained head | [pipeline.py](vico_lite/perception/pipeline.py); the unused projection head is kept as an extension point in [alignment/](vico_lite/alignment/align.py) |
| Memory module — object table, EMA team latent | [memory/hub.py](vico_lite/memory/hub.py) |
| Language module — optional, inactive by default | [language/reasoner.py](vico_lite/language/reasoner.py), [language/guidance.py](vico_lite/language/guidance.py) |
| Policy module — candidate generation and scoring | [policy/policy.py](vico_lite/policy/policy.py) |
| Execution module — TDW commands, A\*, skip/guard feedback | [execution/executor.py](vico_lite/execution/executor.py) |

The candidate score is

```
s(a) = w_d / (d + eps) + (w_n + w_v) * c        w_d=1.5, w_n=0.8, w_v=0.6, eps=1e-3
```

implemented once in `PolicyReasoner._score_candidate` and reused by
`ViCoPolicy._candidate_score`. The weights are set from the task structure, not
tuned; `scripts/ablations/score_weights.sh` sweeps them.

## Layout

```
vico_lite/            the agent stack, one package per method section
  perception/         region proposal, frozen CLIP and Kosmos-2, top-K selection
  memory/             shared object table and team latent (TeamMemoryHub)
  language/           optional LLM proposal branch, off by default
  policy/             deterministic candidate generation and ranking
  execution/          TDW command grounding, A* navigation, failure feedback
  alignment/          optional learned-alignment extension point (unused)
tdw-gym/              gym wrapper, evaluation entry point, baseline agents
  vico_agent.py       binds one replicant to a ViCo-Lite policy
  challenge.py        evaluation loop and command-line interface
  h_agent.py          RHP baseline (upstream)
  lm_agent.py         CoELA baseline (upstream)
transport_challenge_multi_agent/   low-level TDW controller (upstream)
dataset/              scene and task configurations
scripts/              reproduction scripts (see below)
tests/                unit tests that need neither CLIP weights nor the TDW build
docs/                 benchmark reference and upstream API documentation
```

## Setup

```bash
conda create -n vico_lite python=3.9
conda activate vico_lite
pip install -e .
pip install git+https://github.com/openai/CLIP.git
```

On a headless Linux server, follow the
[TDW installation guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md)
to configure the X server, then check the build with `python demo/demo_scene.py`.

Non-oracle perception additionally needs the segmentation model. Install
`mmdetection`:

```bash
pip install -U openmim && mim install mmengine "mmcv>=2.0.0" mmdet
```

and put the fine-tuned Mask R-CNN weights
([download](https://drive.google.com/file/d/1S5id94R2rVeawVrES0eWQzCr_bW81Z3v/view?usp=drive_link))
in `detection_pipeline/` next to its `config.py`. `python detection_pipeline/test_install.py`
verifies the install.

## Reproducing the results

Every script writes to `results/<experiment>/<run_id>/` and prints the average
transport rate when it finishes. `PORT`, `PYTHON`, `OUTPUT_DIR` and
`MAX_FRAMES` can be overridden through the environment.

```bash
bash scripts/eval_oracle.sh              # 24 episodes, ground-truth segmentation
bash scripts/eval_no_oracle.sh           # 24 episodes, segmentation from RGB
bash scripts/ablations/shared_memory.sh  # shared memory / heuristic scoring on and off
bash scripts/ablations/top_k.sh          # sensitivity to the retained candidate count
bash scripts/ablations/perception.sh     # graded perception strength, down to blind
bash scripts/ablations/score_weights.sh  # relevance weights of s(a)
```

The ablation subsets are the leading 12 and 8 episodes of the 24-episode
evaluation order, defined in `scripts/lib.sh`.

Flags that select a configuration, all on `tdw-gym/challenge.py`:

| Flag | Effect |
| --- | --- |
| `--no_gt_mask` | Estimate labels and masks from RGB instead of using the environment's |
| `--seg_dropout P` | Drop a fraction `P` of the segmented objects each frame |
| `--symbolic_top_k K` | Number of scored objects retained per frame |
| `--no_shared_memory` | Give each agent its own memory hub |
| `--no_heuristic_scoring` | Rank candidates by distance alone |
| `--w_d` / `--w_n` / `--w_v` | Override the candidate score weights |
| `--enable_llm_reasoner` | Turn on the optional LLM branch (off in all reported results) |

The LLM branch is off by default: `ViCoConfig.force_heuristics` is `True`, no
OpenAI client is constructed, and no request is issued. Enabling it needs
`OPENAI_API_KEY`, which can be placed in a `.env` file — see `.env.example`.

## Results

Transport rate on the 24-episode TDW-MAT test set. Baseline numbers are as
reported by CoELA; ViCo-Lite makes no LLM call in either setting.

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

Removing shared memory is what costs the most: completion drops from 92% to 30%
under oracle perception and from 83% to 27% without it. Removing heuristic
scoring alone changes neither, which is consistent with a transport task whose
ranking is dominated by distance.

## Tests

```bash
python -m unittest discover -s tests
```

The suite stubs the CLIP encoder, so it runs without model weights, without a
GPU and without the TDW build. It covers region proposal, top-K selection, the
shared object table and the candidate score.

## Acknowledgements

Built on [CoELA / Co-LLM-Agents](https://github.com/UMass-Foundation-Model/Co-LLM-Agents)
(TDW-MAT environment, RHP and CoELA baselines) and on
[transport_challenge_multi_agent](https://github.com/alters-mit/transport_challenge_multi_agent)
by Esther Alter, which is vendored here under its MIT licence. See
[docs/benchmark.md](docs/benchmark.md) for the benchmark description and
[docs/tdw_api/](docs/tdw_api/) for the low-level controller API.
