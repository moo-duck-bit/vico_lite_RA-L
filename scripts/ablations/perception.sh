#!/usr/bin/env bash
# Perception robustness: drop a fraction P of the segmented objects each
# frame, from full perception (P=0) through to blind.
. "$(dirname "$0")/../lib.sh"

for p in 0.0 0.3 0.6 0.9; do
  run_vico ablation_perception "p_$p" "$EPISODES_8" --seg_dropout "$p" --no_save_img "$@"
done
run_vico ablation_perception blind "$EPISODES_8" --no_gt_mask --no_save_img "$@"
