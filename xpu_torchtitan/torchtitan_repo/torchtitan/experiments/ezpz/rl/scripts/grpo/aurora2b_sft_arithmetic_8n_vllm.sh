#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=04:00:00
#PBS -l filesystems=flare:home
#PBS -l select=10
#PBS -q workq
#PBS -j oe
#
# 8N Sunspot production GRPO — vLLM server-mode variant of
# `aurora2b_sft_arithmetic_8n.sh`.
#
# Layout:
#   - 1 dedicated XPU tile on the head node runs `trl vllm-serve` from
#     `venvs/vllm-test/` (torch 2.12 + vllm-xpu). It hosts the SFT'd
#     `checkpoint-729-hf` and exposes an OpenAI-compat API on :8000.
#   - The remaining 95 tiles run the GRPO trainer in .venv/ (torch 2.13
#     + TRL). The trainer talks to the vLLM server via
#     --use_vllm --vllm_mode=server --vllm_server_base_url=...
#
# Goal: replace the all-ranks-`.generate()` HF path with the vLLM
# OpenAI-API path. Expected ~3–5× speedup on the generation phase at
# EP=12 / matching shapes (numbers TBD on first run).
#
# Pre-merge sanity: confirm `venvs/vllm-test/` stack still works
# standalone (Phase 1 of the wiring plan) before relying on this
# script. See docs/rl/vllm-xpu-wiring-plan.md.
#
# Output: outputs/grpo/aurora2b-sft-arithmetic-8n-vllm/

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }
test -d venvs/vllm-test || { echo "FATAL: venvs/vllm-test/ missing; see docs/rl/vllm-xpu-investigation.md"; exit 1; }

MODEL=outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf
CKPT_DIR=outputs/grpo/aurora2b-sft-arithmetic-8n-vllm
LOG_DIR="logs/grpo-aurora2b-sft-arithmetic-8n-vllm-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

HEAD_NODE="$(head -1 "${PBS_NODEFILE}")"
VLLM_PORT=8765                   # avoid clashes with TRL's :8000 default
VLLM_URL="http://${HEAD_NODE}:${VLLM_PORT}"

echo "=== 8N GRPO + vLLM server: SFT'd AuroraGPT-2B-tulu-mix + arithmetic ===" \
    | tee "${LOG_DIR}/run.log"
echo "Allocation: select=10  HEAD_NODE=${HEAD_NODE}  VLLM_URL=${VLLM_URL}" \
    | tee -a "${LOG_DIR}/run.log"
echo "Model: ${MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "Output: ${CKPT_DIR}" | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# --- step 1: launch vLLM server on a dedicated tile of the head node ----
# `ezpz launch --np 1 --hosts ${HEAD_NODE}` runs on a single tile;
# `--cpu-bind=none` lets the vLLM workers pick their own affinity.
echo "[$(date +%T)] launching vllm server on ${HEAD_NODE}:${VLLM_PORT}" \
    | tee -a "${LOG_DIR}/run.log"
MODEL="${MODEL}" PORT="${VLLM_PORT}" TP=1 GPU_MEM_UTIL=0.5 \
    nohup torchtitan/experiments/ezpz/rl/scripts/vllm_serve_xpu.sh \
    > "${LOG_DIR}/vllm_serve.log" 2>&1 &
VLLM_PID=$!
echo "vllm PID=${VLLM_PID}" | tee -a "${LOG_DIR}/run.log"

# Trap to kill the server on trainer exit (normal OR crash).
trap 'echo "[$(date +%T)] cleaning up vllm pid=${VLLM_PID}"; kill -TERM ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT INT TERM

# Wait for server health — up to 5 min (model load + KV cache alloc).
echo "[$(date +%T)] waiting for ${VLLM_URL}/health ..." | tee -a "${LOG_DIR}/run.log"
SECONDS_WAITED=0
until curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FATAL: vllm server died before becoming healthy. See ${LOG_DIR}/vllm_serve.log" \
            | tee -a "${LOG_DIR}/run.log"
        exit 1
    fi
    if (( SECONDS_WAITED >= 300 )); then
        echo "FATAL: vllm server not healthy after 300s. See ${LOG_DIR}/vllm_serve.log" \
            | tee -a "${LOG_DIR}/run.log"
        exit 1
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
done
echo "[$(date +%T)] vllm server healthy after ${SECONDS_WAITED}s" \
    | tee -a "${LOG_DIR}/run.log"

# --- step 2: launch trainer on the remaining 95 tiles ------------------
# Tiles 1-11 on the head node + all 12 tiles on each of the other 9 nodes
# = 11 + 9*12 = 119 tiles. We grab 95 (8 train × 12 - 1) for the trainer
# and ezpz_setup_job will hand us those via the hostfile.
#
# Note --use_vllm --vllm_mode=server pass-through: these are TRL
# GRPOConfig fields that EzpzGRPOConfig inherits unchanged. TRL handles
# the OpenAI-API client wiring internally.
ezpz launch --np 95 -ppn 12 --auto-retry --max-failover-retries 2 \
    python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task arithmetic \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --max_completion_length 64 \
    --temperature 0.7 \
    --max_steps 1000 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 5 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url "${VLLM_URL}" \
    --vllm_server_timeout 240 \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
