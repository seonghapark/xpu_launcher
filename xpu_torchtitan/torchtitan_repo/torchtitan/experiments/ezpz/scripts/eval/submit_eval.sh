#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -j oe

# ---- Environment ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export TORCH_CPP_LOG_LEVEL=ERROR

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate the venv with lm_eval + vllm
source .venv/bin/activate

# ---- Configuration ----
# Override these via qsub -v "MODEL=2b,STEP=5000,TASKS=hellaswag"
MODEL="${MODEL:-2b}"
STEP="${STEP:-5000}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"

echo "============================================"
echo "AGPT Evaluation Job"
echo "============================================"
echo "PBS_JOBID: ${PBS_JOBID}"
echo "MODEL:     ${MODEL}"
echo "STEP:      ${STEP}"
echo "TASKS:     ${TASKS}"
echo "NNODES:    $(wc -l < "${PBS_NODEFILE}")"
echo "============================================"

# Run the eval pipeline
bash torchtitan/experiments/ezpz/scripts/eval/convert_and_eval.sh \
    --model "${MODEL}" \
    --step "${STEP}" \
    --tasks "${TASKS}" \
    "$@"
