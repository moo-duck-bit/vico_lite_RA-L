#!/usr/bin/env bash
# Main result, oracle perception: ground-truth segmentation, 24 test episodes.
. "$(dirname "$0")/lib.sh"

run_vico vico_lite "${RUN_ID:-oracle}" "$EPISODES_24" --no_save_img "$@"
