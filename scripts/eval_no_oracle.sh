#!/usr/bin/env bash
# Main result, non-oracle perception: object labels and masks are estimated
# from the egocentric RGB frame by the detector in detection_pipeline/, the
# same one the CoELA and RHP baselines use.
. "$(dirname "$0")/lib.sh"

run_vico vico_lite "${RUN_ID:-no_oracle}" "$EPISODES_24" --no_gt_mask --no_save_img "$@"
