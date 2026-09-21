#!/usr/bin/env bash
# Single episode at a large viewport, for screen recording. Keeps the frame
# dumps and stops after the first successful delivery.
. "$(dirname "$0")/lib.sh"

SCREEN_SIZE=1024

run_vico vico_demo "${RUN_ID:-demo}" "${EPISODES:-0}" --stop_on_first_success "$@"
