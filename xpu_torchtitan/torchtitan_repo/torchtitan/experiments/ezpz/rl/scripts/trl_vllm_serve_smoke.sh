#!/bin/bash --login
#PBS -A datascience
#PBS -N trl-vllm-serve
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# TRL `vllm-serve` smoke for ezpz/rl Track C (TRL vllm_mode="server").
#
# What it does:
#   1. Launches `trl vllm-serve` against a small HF checkpoint from the
#      unified venvs/rl-vllm/ venv (py3.12 + torch 2.12+xpu + vllm 0.22
#      + triton-xpu 3.7.1 + TRL 1.6).
#   2. Polls /health/ until the server is up (or 5 min timeout).
#   3. POSTs a sanity completion request, checks the response.
#   4. Tears down the server.
#
# Applies all the env scrubs and tweaks discovered during the bare-vLLM
# smoke debug chain (12468740..12468754). Specifically: unsets the
# CCL_*/FI_* vars from ezpz_setup_env that contaminate vLLM's
# EngineCore subprocess.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# Don't route our own server health/generate polls through the ALCF
# proxy — it can't reach loopback. The model+pip downloads above use
# the proxy normally.
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# Scrub the CCL_*/FI_* env vars that ezpz_setup_env exports.
# vLLM's EngineCore subprocess has no MPI bootstrap and needs the
# oneCCL defaults; the cxi OFI provider in particular needs a Slingshot
# NIC handle that only mpiexec-bootstrapped processes have.
unset CCL_OP_SYNC CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_OFI_PROVIDER
unset FI_PROVIDER FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE

# Default to the small Qwen3-0.6B checkpoint we already have on disk
# (downloaded for the upstream rl/ smoke).
MODEL="${MODEL:-${SUBMIT_DIR}/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B}"
PORT="${PORT:-8765}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/trl-vllm-serve-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"
SERVE_LOG="${LOG_DIR}/vllm_serve.log"

echo "=== trl vllm-serve smoke ===" | tee "${LOG}"
echo "MODEL=${MODEL}" | tee -a "${LOG}"
echo "PORT=${PORT}  TP=${TP}  GPU_MEM_UTIL=${GPU_MEM_UTIL}" | tee -a "${LOG}"
echo "HOST=$(hostname -s)" | tee -a "${LOG}"
date | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# --- Phase 1: stack import sanity ----------------------------------
echo "[$(date +%T)] Phase 1: stack import sanity" | tee -a "${LOG}"
"${SUBMIT_DIR}/venvs/rl-vllm/bin/python" -c "
import torch, vllm, trl, transformers
print(f'  python  : {__import__(\"sys\").version.split()[0]}')
print(f'  torch   : {torch.__version__}')
print(f'  vllm    : {vllm.__version__}')
print(f'  trl     : {trl.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  xpu_count: {torch.xpu.device_count()}')
" 2>&1 | tee -a "${LOG}" || { echo "FATAL Phase 1: stack broken"; exit 1; }

# --- Phase 2: trl vllm-serve --------------------------------------
echo "" | tee -a "${LOG}"
echo "[$(date +%T)] Phase 2: launching trl vllm-serve on :${PORT}" | tee -a "${LOG}"

nohup "${SUBMIT_DIR}/venvs/rl-vllm/bin/trl" vllm-serve \
    --model "${MODEL}" \
    --tensor_parallel_size "${TP}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --enforce_eager \
    > "${SERVE_LOG}" 2>&1 &
SERVE_PID=$!

trap 'echo "[cleanup] kill -TERM ${SERVE_PID}"; kill -TERM ${SERVE_PID} 2>/dev/null || true; sleep 2; kill -KILL ${SERVE_PID} 2>/dev/null || true' EXIT

echo "serve PID=${SERVE_PID}" | tee -a "${LOG}"
echo "[$(date +%T)] waiting for /health/ on :${PORT} (up to 300s)..." | tee -a "${LOG}"

SECONDS_WAITED=0
until curl -sf "http://127.0.0.1:${PORT}/health/" > /dev/null 2>&1; do
    if ! kill -0 "${SERVE_PID}" 2>/dev/null; then
        echo "FATAL: serve PID ${SERVE_PID} died before /health/ came up" | tee -a "${LOG}"
        echo "--- last 50 lines of vllm_serve.log ---" | tee -a "${LOG}"
        tail -50 "${SERVE_LOG}" | tee -a "${LOG}"
        echo "VERDICT: FAILED at startup" | tee -a "${LOG}"
        exit 1
    fi
    if (( SECONDS_WAITED >= 300 )); then
        echo "FATAL: /health/ not responding after 300s" | tee -a "${LOG}"
        tail -50 "${SERVE_LOG}" | tee -a "${LOG}"
        echo "VERDICT: FAILED at startup" | tee -a "${LOG}"
        exit 1
    fi
    sleep 3
    SECONDS_WAITED=$((SECONDS_WAITED + 3))
done
echo "[$(date +%T)] /health/ up after ${SECONDS_WAITED}s" | tee -a "${LOG}"

# --- Phase 3: /generate sanity request -----------------------------
echo "" | tee -a "${LOG}"
echo "[$(date +%T)] Phase 3: POST /generate sanity request" | tee -a "${LOG}"
curl -sS -w "\n[HTTP %{http_code}, %{time_total}s]\n" \
     -X POST "http://127.0.0.1:${PORT}/generate/" \
     -H "Content-Type: application/json" \
     -d '{
            "prompts": ["What is 3 + 7 + 2?"],
            "n": 1,
            "temperature": 0,
            "max_tokens": 32
         }' 2>&1 | tee -a "${LOG}"

echo "" | tee -a "${LOG}"
echo "VERDICT: complete" | tee -a "${LOG}"
