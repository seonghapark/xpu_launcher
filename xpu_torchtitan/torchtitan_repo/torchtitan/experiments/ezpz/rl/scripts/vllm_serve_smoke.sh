#!/bin/bash --login
#PBS -A datascience
#PBS -N vllm-xpu-serve-smoke
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Standalone smoke for the vLLM-XPU server stack used by ezpz/rl.
# Phase 1 + Phase 2 of `docs/rl/vllm-xpu-wiring-plan.md`:
#
#   - Phase 1: confirm `venvs/vllm-test/` torch-2.12 + vllm-xpu stack
#     still loads the SFT'd `checkpoint-729-hf` model on a fresh
#     Sunspot allocation (the original verification was 2026-06-10;
#     the env has shifted under us before).
#   - Phase 2: bring up `trl vllm-serve` on :8765, wait for /health,
#     POST a sanity completion request, tear down.
#
# Exit 0 if both pass, non-zero (and tail of vllm log) if either fails.
# Doesn't touch any production checkpoint dir.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

MODEL="${MODEL:-outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf}"
PORT="${PORT:-8765}"
JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/vllm-xpu-smoke-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "=== vLLM-XPU smoke ===" | tee "${LOG_DIR}/run.log"
echo "MODEL=${MODEL}  PORT=${PORT}  HOST=$(hostname -s)" | tee -a "${LOG_DIR}/run.log"

# --- Phase 1: bare vllm-test stack ----------------------------------
echo "" | tee -a "${LOG_DIR}/run.log"
echo "[$(date +%T)] Phase 1: import vllm + torch from venvs/vllm-test" \
    | tee -a "${LOG_DIR}/run.log"
venvs/vllm-test/bin/python -c "
import torch, vllm
print(f'  torch={torch.__version__}  vllm={vllm.__version__}')
print(f'  xpu_count={torch.xpu.device_count() if hasattr(torch, \"xpu\") else 0}')
" 2>&1 | tee -a "${LOG_DIR}/run.log" || { echo "FATAL Phase 1: vllm-test stack broken"; exit 1; }

# --- Phase 2: trl vllm-serve over the SFT'd checkpoint --------------
echo "" | tee -a "${LOG_DIR}/run.log"
echo "[$(date +%T)] Phase 2: launching trl vllm-serve on :${PORT}" \
    | tee -a "${LOG_DIR}/run.log"

MODEL="${MODEL}" PORT="${PORT}" TP=1 GPU_MEM_UTIL=0.5 \
    nohup torchtitan/experiments/ezpz/rl/scripts/vllm_serve_xpu.sh \
    > "${LOG_DIR}/vllm_serve.log" 2>&1 &
VLLM_PID=$!
trap 'echo "cleanup: kill -TERM ${VLLM_PID}"; kill -TERM ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

echo "vllm PID=${VLLM_PID}" | tee -a "${LOG_DIR}/run.log"
echo "[$(date +%T)] waiting for /health (up to 300s)..." | tee -a "${LOG_DIR}/run.log"

SECONDS_WAITED=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "FATAL: vllm server died before becoming healthy" \
            | tee -a "${LOG_DIR}/run.log"
        echo "--- last 30 lines of vllm_serve.log ---" \
            | tee -a "${LOG_DIR}/run.log"
        tail -30 "${LOG_DIR}/vllm_serve.log" | tee -a "${LOG_DIR}/run.log"
        exit 1
    fi
    if (( SECONDS_WAITED >= 300 )); then
        echo "FATAL: not healthy after 300s" | tee -a "${LOG_DIR}/run.log"
        tail -30 "${LOG_DIR}/vllm_serve.log" | tee -a "${LOG_DIR}/run.log"
        exit 1
    fi
    sleep 5
    SECONDS_WAITED=$((SECONDS_WAITED + 5))
done
echo "[$(date +%T)] healthy after ${SECONDS_WAITED}s" | tee -a "${LOG_DIR}/run.log"

# Sanity completion
echo "" | tee -a "${LOG_DIR}/run.log"
echo "[$(date +%T)] POST /v1/completions sanity request" | tee -a "${LOG_DIR}/run.log"
curl -sS -X POST "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"What is 3 + 7 + 2?\", \"max_tokens\": 32, \"model\": \"${MODEL}\"}" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "[$(date +%T)] VERDICT: smoke passed" | tee -a "${LOG_DIR}/run.log"
