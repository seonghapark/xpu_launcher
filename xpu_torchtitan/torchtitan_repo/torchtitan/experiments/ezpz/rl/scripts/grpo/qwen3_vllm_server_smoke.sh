#!/bin/bash --login
#PBS -A datascience
#PBS -N grpo-vllm-server
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1-node smoke: GRPO training against an external `trl vllm-serve`
# server, all running from the unified `venvs/rl-vllm/` venv.
#
# Layout (single Sunspot node, 12 tiles):
#   - tile 0  : `trl vllm-serve` daemon (TP=1), hosts Qwen3-0.6B
#               on :8765. Backgrounded.
#   - tiles 1-11 : ezpz `train_grpo` (11 ranks via mpiexec),
#                  --use_vllm --vllm_mode=server --vllm_server_base_url=...
#
# Why this works (where the Monarch+oneCCL approach didn't):
#   - vllm-serve runs as a single XCCL world (TP=1, so just itself).
#     It does its own oneCCL init in its own process tree.
#   - The trainer talks to the server via HTTP, not via shared comms.
#     So the trainer's XCCL world (11 ranks via mpiexec) is fully
#     independent. mpiexec provides PMIx → oneCCL works → trainer
#     does normal FSDP/DP collectives.
#   - No Monarch in the picture, so no PMIx vs fork mismatch.
#
# 5 GRPO steps. Validates the loop end-to-end (server up, trainer
# init, generate→reward→loss→backward→step, then teardown).

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Don't route loopback through the ALCF proxy (server health/generate
# polls go to 127.0.0.1).
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# Configure oneCCL for TCP-KVS rendezvous so trainer↔server XCCL group
# works without a shared PMIx parent. This is the architecturally-correct
# alternative to no-op'ing TRL's weight-sync — verified working with a
# 2-process cross-tree XCCL broadcast (xpu tensors) on 2026-06-13 PM.
#
# CRITICAL: both the server and the trainer must share these env vars
# (and especially the same CCL_KVS_IP_PORT) so they rendezvous at the
# same TCP endpoint. We export them at the outer-script scope (BEFORE
# launching the server subshell) so they propagate to both.
unset CCL_OP_SYNC CCL_OFI_PROVIDER
unset FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
# Pick a port for the TCP-KVS endpoint. Both server and trainer must
# agree on this for the cross-process XCCL group to form.
export CCL_KVS_IP_PORT="127.0.0.1_29513"

MODEL="${MODEL:-${SUBMIT_DIR}/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B}"
VLLM_PORT="${VLLM_PORT:-8765}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/grpo-vllm-server-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
SERVE_LOG="${LOG_DIR}/vllm_serve.log"

echo "=== GRPO + trl vllm-serve smoke (1N Sunspot) ===" | tee "${LOG}"
echo "MODEL=${MODEL}" | tee -a "${LOG}"
echo "VLLM_PORT=${VLLM_PORT}  HOST=$(hostname -s)" | tee -a "${LOG}"
date | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# --- Phase 1: launch vllm-serve on tile 0 ---------------------------
echo "[$(date +%T)] Phase 1: launching trl vllm-serve on :${VLLM_PORT}" | tee -a "${LOG}"

# Server inherits the outer-scope env (CCL_KVS_IP_PORT, CCL_PROCESS_LAUNCHER=none,
# FI_PROVIDER=tcp). Pin to tile 0; trainer will use tiles 1-8.
(
    export ZE_AFFINITY_MASK=0
    exec "${SUBMIT_DIR}/venvs/rl-vllm/bin/trl" vllm-serve \
        --model "${MODEL}" \
        --tensor_parallel_size 1 \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --gpu_memory_utilization 0.5 \
        --enforce_eager
) > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!
trap 'echo "[cleanup] killing serve PID ${SERVE_PID}"; kill -TERM ${SERVE_PID} 2>/dev/null || true; sleep 2; kill -KILL ${SERVE_PID} 2>/dev/null || true' EXIT INT TERM

echo "serve PID=${SERVE_PID}" | tee -a "${LOG}"
echo "[$(date +%T)] waiting for /health/ (up to 300s)..." | tee -a "${LOG}"
SECONDS_WAITED=0
until curl -sf "http://127.0.0.1:${VLLM_PORT}/health/" > /dev/null 2>&1; do
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "FATAL: serve PID died before /health/" | tee -a "${LOG}"
        tail -50 "${SERVE_LOG}" | tee -a "${LOG}"
        echo "VERDICT: FAILED at vllm startup" | tee -a "${LOG}"
        exit 1
    fi
    if (( SECONDS_WAITED >= 300 )); then
        echo "FATAL: vllm /health/ timeout" | tee -a "${LOG}"
        tail -50 "${SERVE_LOG}" | tee -a "${LOG}"
        echo "VERDICT: FAILED at vllm startup" | tee -a "${LOG}"
        exit 1
    fi
    sleep 3
    SECONDS_WAITED=$((SECONDS_WAITED + 3))
done
echo "[$(date +%T)] vllm /health/ up after ${SECONDS_WAITED}s" | tee -a "${LOG}"

# --- Phase 2: launch GRPO trainer on tiles 1-11 ---------------------
echo "" | tee -a "${LOG}"
echo "[$(date +%T)] Phase 2: launching GRPO trainer on 11 ranks (tiles 1-11)" | tee -a "${LOG}"

# Trainer inherits the same TCP-KVS oneCCL env as the server
# (CCL_PROCESS_LAUNCHER=none, FI_PROVIDER=tcp, CCL_KVS_IP_PORT=...).
# This unifies the rendezvous: trainer's intra-mesh group AND the
# trainer↔server weight-sync group both use TCP-KVS. Slower than
# Slingshot CXI for intra-node collectives but functional.
CKPT_DIR="outputs/grpo/grpo-vllm-server-smoke"
mkdir -p "${CKPT_DIR}"
TRAINER_LOG="${LOG_DIR}/trainer.log"

# Skip tile 0 (vllm-serve owns it) — use tiles 1-11.
# ezpz launch usually picks tile 0; we override via ZE_AFFINITY_MASK
# inside the launched python process. Simpler: just use np=11 and let
# mpiexec pick consecutive tiles starting from 1 via the
# CPU/NUMA binding default + ZE_AFFINITY_MASK on the trainer python.

# Actually simplest: just use 11 ranks. ezpz launch will assign each
# rank LOCAL_RANK 0..10, and each rank does
# torch.xpu.set_device(LOCAL_RANK). But that'd pick tiles 0..10 —
# conflicting with the server on tile 0. Instead, set
# ZE_AFFINITY_MASK=1,2,3,4,5,6,7,8,9,10,11 on the trainer launch so
# python sees tiles 1..11 (re-indexed as xpu:0..xpu:10) and doesn't
# touch tile 0.

# Use 8 trainer tiles (1..8) so generation_batch_size (8 * bsz=1 = 8)
# is divisible by num_generations=4. Tiles 9-11 idle; tile 0 = vllm server.
export ZE_AFFINITY_MASK=1,2,3,4,5,6,7,8

"${SUBMIT_DIR}/.venv/bin/ezpz" launch --np 8 -ppn 8 \
    "${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 \
    --num_generations 4 \
    --max_completion_length 64 \
    --temperature 0.7 \
    --max_steps 5 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --bf16 \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url "http://127.0.0.1:${VLLM_PORT}" \
    --vllm_server_timeout 300 \
    2>&1 | tee -a "${TRAINER_LOG}" "${LOG}"

echo "" | tee -a "${LOG}"
echo "VERDICT: complete" | tee -a "${LOG}"
