#!/usr/bin/env bash
# Sensitivity to the number of scored objects retained per frame.
. "$(dirname "$0")/../lib.sh"

for k in 1 3 5 8; do
  run_vico ablation_topk "k_$k" "$EPISODES_8" --symbolic_top_k "$k" --no_save_img "$@"
done
# No-ranking control: keep every visible object, never rank them.
run_vico ablation_topk no_ranking "$EPISODES_8" --symbolic_top_k 999 --no_save_img "$@"
