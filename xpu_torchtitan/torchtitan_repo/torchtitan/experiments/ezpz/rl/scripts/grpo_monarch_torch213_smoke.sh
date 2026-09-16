#!/bin/bash --login
#PBS -A datascience
#PBS -N grpo-monarch-t213
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Monarch + vLLM GRPO smoke on torch 2.13 + py3.13.
#
# Breakthrough on 2026-06-14: torch 2.13's xpu allocator produces
# SYCL-USM-device-typed tensors that DTensor's mesh_broadcast can use
# under both mpiexec AND Monarch's spawn_procs. (torch 2.12 + Monarch
# failed every broadcast at ccl_check_usm_pointers.)
#
# Stack: venvs/rl-monarch-torch213/ (py3.13)
#   - torch 2.13.0.dev20260519+xpu + triton-xpu 3.7.1
#   - torchmonarch 0.5.0 + torchstore HEAD
#   - vllm 0.22.1 + vllm-xpu-kernels (rebuilt from source against
#     torch 2.13; the prebuilt 0.1.9.1 wheel can't link because it
#     needs c10::impl::cow::materialize_cow_storage which 2.12 had
#     and 2.13 dropped)
#   - trl 1.6 + transformers 5.12 + accelerate 1.14 + datasets 5.0

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# Use ONLY level_zero (not opencl) — opencl backend can race with L0
# when multiple processes share the same physical tile and trip
# oneDNN's "could not create a memory" during alloc.
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# TCP-KVS oneCCL env (no PMIx required) — same fix that landed
# Track C TRL+vllm-serve. Monarch actors get these via
# setup_oneccl_tcp_kvs_for_xpu() in apply_all_xpu_patches.
unset CCL_OP_SYNC CCL_OFI_PROVIDER
unset FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=tcp
export CCL_KVS_IP_PORT="127.0.0.1_29635"
# Try disabling the vllm-xpu-kernels custom op impls — falling back to
# torch-native (oneDNN). Possibly our custom-built vllm-xpu-kernels
# .so files don't play nicely with torch 2.13's allocator and produce
# "could not create a memory" failures during profile_run.
export VLLM_DISABLED_KERNELS=xpu_kernels
export CCL_LOG_LEVEL=warn
# vLLM-XPU's selector raises ValueError when no attention backend is
# explicitly selected. Pick FLASH_ATTN (vllm-xpu has its own flash
# kernel; FA2 path is the supported path on XPU).
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

export PYTHONPATH="${SUBMIT_DIR}:${PYTHONPATH:-}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/grpo-monarch-t213-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/run.log"

echo "=== upstream rl/ GRPO via Monarch on torch 2.13 ===" | tee "${LOG}"
date | tee -a "${LOG}"
"${SUBMIT_DIR}/venvs/rl-monarch-torch213/bin/python" -c "
import torch, monarch, vllm_xpu_kernels, vllm, trl
print('torch', torch.__version__)
print('triton-xpu', __import__('triton').__version__)
print('monarch OK')
print('vllm', vllm.__version__)
print('trl', trl.__version__)
print('xpu_count', torch.xpu.device_count())
" 2>&1 | tee -a "${LOG}"
echo "" | tee -a "${LOG}"

# Plain python — Monarch picks tiles itself via this_host().spawn_procs(...)
# and our EzpzPerHostProvisioner partitions them with ZE_AFFINITY_MASK.
"${SUBMIT_DIR}/venvs/rl-monarch-torch213/bin/python" \
    -m torchtitan.experiments.ezpz.rl.train_upstream \
    --module rl --config rl_grpo_qwen3_0_6b_varlen \
    --hf_assets_path "${SUBMIT_DIR}/torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B" \
    --trainer.debug.seed 42 \
    --trainer.parallelism.tensor-parallel-degree 1 \
    --generator.parallelism.tensor-parallel-degree 1 \
    --generator.gpu-memory-limit 0.5 \
    --generator.sampling.max-tokens 256 \
    --generator.cudagraph.no-enable \
    --compile.no-enable \
    --batcher.batch.seq-len 512 \
    2>&1 | tee -a "${LOG}"
# TP=1 everywhere to dodge XCCL cross-rank collectives — those crash
# with oneCCL ccl_check_usm_pointers on Monarch-spawned actors. (XCCL
# works fine under mpiexec via PMIx; the bug is specific to actor
# processes that bypass PMIx and use TCP-KVS rendezvous.)
# With TP=1, trainer = 1 rank, generator = 1 rank, no intra-mesh
# collectives needed.

echo "" | tee -a "${LOG}"
echo "VERDICT: complete" | tee -a "${LOG}"
