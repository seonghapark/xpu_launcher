#!/bin/bash --login
#PBS -A datascience
#PBS -N grpo-qwen3-smoke
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# First end-to-end attempt at upstream `torchtitan.experiments.rl.train`
# on Sunspot XPU, via the ezpz mirror entrypoint
# `torchtitan.experiments.ezpz.rl.train_upstream` which applies XPU
# compatibility patches before importing upstream.
#
# Config: `rl_grpo_qwen3_0_6b_varlen` (Qwen3-0.6B, 6 GPUs total = 2
# trainer + 4 generator, 10 steps, alphabet-sort env).
#
# Verifies on Sunspot: Monarch HostMesh + spawn_procs with
# ZE_AFFINITY_MASK partitioning, vLLM generator init, TorchStore
# weight pull, GRPO training step.
#
# Expect this to crash on the first few attempts — each crash maps
# to one of the XPU porting points we may have missed. See
# `xpu_overrides.py` for what's already patched.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# train_upstream.py also pops these on the controller side, but
# stripping in the launcher is belt-and-braces (and matches the
# bare-smoke pattern).
unset CCL_OP_SYNC CCL_OFI_PROVIDER
unset FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE

# TCP-KVS XCCL rendezvous — same fix that landed the TRL vllm-serve
# GRPO smoke (job 12468780). With CCL_PROCESS_LAUNCHER=none + TCP
# fabric + a shared CCL_KVS_IP_PORT, oneCCL forms XCCL groups across
# Monarch-spawned actor processes without needing a PMIx parent.
#
# Each Monarch actor's _bootstrap also calls
# setup_oneccl_tcp_kvs_for_xpu() (via apply_all_xpu_patches), but it
# only sets the static knobs. The PORT must be set here so every
# actor inherits the same value.
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_KVS_IP_PORT="127.0.0.1_29515"

# Upstream `rl/` imports `torchtitan.*` from the source tree.
export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH:-}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/grpo-qwen3-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "=== upstream rl/train via ezpz mirror, Qwen3-0.6B GRPO ===" | tee "${LOG_DIR}/run.log"
date | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Plain python — the Monarch controller picks tiles itself via
# `this_host().spawn_procs(...)` and `EzpzPerHostProvisioner`
# partitions them with ZE_AFFINITY_MASK. Each spawned actor's
# _bootstrap calls apply_all_xpu_patches(), which includes
# setup_oneccl_tcp_kvs_for_xpu() — so every actor inherits the
# TCP-KVS rendezvous config and joins the same XCCL world.
"${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_upstream \
    --module rl --config rl_grpo_qwen3_0_6b_varlen \
    --hf_assets_path "${SUBMIT_DIR}/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B" \
    --trainer.debug.seed 42 \
    2>&1 | tee -a "${LOG_DIR}/run.log"
# --trainer.debug.seed 42: skip seed-broadcast branch in set_determinism
#   (originally a workaround for oneCCL misbehaving on broadcasts; now
#   a reproducibility win).
#
# Default config: trainer TP=2 + generator TP=4 = 6 tiles total.

echo "" | tee -a "${LOG_DIR}/run.log"
echo "VERDICT: complete" | tee -a "${LOG_DIR}/run.log"
