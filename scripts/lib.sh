#!/usr/bin/env bash
# Shared setup for the ViCo-Lite run scripts.
#
# Every script sources this file, then calls run_vico with the flags that
# distinguish it. Paths are resolved from the repository root, so the scripts
# work from any working directory and on any machine.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
PORT="${PORT:-1071}"
SCREEN_SIZE="${SCREEN_SIZE:-512}"
MAX_FRAMES="${MAX_FRAMES:-3000}"
DATA_PREFIX="${DATA_PREFIX:-dataset/dataset_test/}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# The 24 test episodes, in the evaluation order used by CoELA. The 12- and
# 8-episode ablation subsets are the leading episodes of this same list.
EPISODES_24="0 11 17 18 1 2 3 21 22 23 4 5 6 7 8 9 10 12 13 14 15 16 19 20"
EPISODES_12="0 11 17 18 1 2 3 21 22 23 4 5"
EPISODES_8="0 11 17 18 1 2 3 21"

# Load OPENAI_API_KEY only if the optional LLM branch is going to be used.
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_ROOT/.env"
  set +a
fi

free_port() {
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti "tcp:$PORT" 2>/dev/null || true)"
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  fi
}

# run_vico <experiment_name> <run_id> <episodes> [extra challenge.py flags...]
run_vico() {
  local experiment="$1"; shift
  local run_id="$1"; shift
  local episodes="$1"; shift

  echo "[$experiment/$run_id] episodes: $episodes"
  free_port
  trap free_port EXIT

  "$PYTHON" tdw-gym/challenge.py \
    --agents vico_agent vico_agent \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name "$experiment" \
    --run_id "$run_id" \
    --data_path test_env.json \
    --data_prefix "$DATA_PREFIX" \
    --eval_episodes $episodes \
    --max_frames "$MAX_FRAMES" \
    --port "$PORT" \
    --screen_size "$SCREEN_SIZE" \
    --communication \
    --no_save_img \
    "$@"
}
