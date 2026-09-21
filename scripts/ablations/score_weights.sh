#!/usr/bin/env bash
# Sweep the relevance weights of s(a) = w_d / (d + eps) + (w_n + w_v) * c,
# holding w_d fixed.
. "$(dirname "$0")/../lib.sh"

run_vico ablation_weights default "$EPISODES_8" --no_save_img "$@"
run_vico ablation_weights zero    "$EPISODES_8" --w_n 0.0 --w_v 0.0 --no_save_img "$@"
run_vico ablation_weights x4      "$EPISODES_8" --w_n 3.2 --w_v 2.4 --no_save_img "$@"
