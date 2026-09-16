#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi

MODEL="${MODEL:-2b}"

_fallback_dfl="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"
DFL="${DFL:-${DATA_FILE_LIST:-${_fallback_dfl}}}"

MODULE=${MODULE:-"ezpz.agpt"}
CONFIG=${CONFIG:-"ezpz_agpt_${MODEL}"}

NGPU=${NGPU:-${NGPUS:-${WORLD_SIZE:-4}}}
COMM_MODE=${COMM_MODE:-""}

export LOG_RANK=${LOG_RANK:-0}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

CHECKPOINT_DIR="aGPT-${MODEL}-ws${NGPU}-$(basename "${DFL}")"

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.experiments.ezpz.train \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        "$@" \
        --comm.mode="${COMM_MODE}" \
        --training.steps 1
else
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --debug.print_config \
        --module "${MODULE}" \
        --config "${CONFIG}" \
        --training.dataset_path "${DFL}" \
        --checkpoint.folder "${CHECKPOINT_DIR}" \
        "$@"
fi
