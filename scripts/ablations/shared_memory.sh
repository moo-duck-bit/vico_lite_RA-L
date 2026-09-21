#!/usr/bin/env bash
# Core module ablation: shared memory and heuristic scoring, on and off.
# Run under oracle perception by default; pass --no_gt_mask for the
# non-oracle half of the table.
. "$(dirname "$0")/../lib.sh"

run_vico ablation_core full           "$EPISODES_12" --no_save_img "$@"
run_vico ablation_core no_shared_mem  "$EPISODES_12" --no_shared_memory --no_save_img "$@"
run_vico ablation_core no_heuristic   "$EPISODES_12" --no_heuristic_scoring --no_save_img "$@"
run_vico ablation_core neither        "$EPISODES_12" --no_shared_memory --no_heuristic_scoring --no_save_img "$@"
