#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=36
#PBS -q workq
#PBS -j oe
#
# 32N (train) + 4N (spare) SFT resume with ezpz launch --auto-retry.
# Replays the 12468220 resume but uses ezpz's bad-node failover so a
# transient ccl::v1::exception (which knocked out 12468218 AND 12468220
# at ~14min and ~10min wall respectively) triggers a node swap +
# auto-restart from the latest checkpoint instead of killing the run.
#
# Why 4 spares: the previous two runs each lost ONE rank to SIGABRT.
# 4 spares lets us survive 4 bad-node events across the remaining
# ~356 steps. Probably overkill, but cheap.
#
# Recognized bad-node patterns ezpz/launch_autoretry.py:_CRASH_PATTERNS_RX:
#   gloo "Connection closed by peer" / "Timed out waiting"
#   OutOfMemoryError
#   UR_RESULT_ERROR_OUT_OF_RESOURCES
#   "died from signal" (NOT cascaded signal 11/15 — those are stripped)
#   EOFError: No data left in file
#
# Our failure signature: "died from signal 6" after a ccl::v1::exception
# — matches "died from signal" and is NOT in the innocent-cascade strip.

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

CKPT_DIR="outputs/sft/aurora2b-sophiag-metamathqa-32n"
RESUME_FROM="${CKPT_DIR}/checkpoint-400"

if [[ ! -d "${RESUME_FROM}" ]]; then
    echo "FATAL: resume checkpoint missing: ${RESUME_FROM}"
    exit 1
fi

LOG_DIR="logs/sft-aurora2b-metamathqa-32n-autoretry-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 32N SFT RESUME (auto-retry, 4 spare nodes) from checkpoint-400 ===" \
    | tee "${LOG_DIR}/run.log"
echo "Allocation: select=36 (32 train + 4 spare)" | tee -a "${LOG_DIR}/run.log"
echo "Failover: --auto-retry --max-failover-retries 3 --spare-nodes 4" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# NOTE: --np 384 = 32 train nodes * 12 ranks/node. --auto-retry needs
# --nproc (rank count) explicitly, not --nhost. The spare pool is
# auto-derived from select=36 - ceil(384/12) = 4 spares.
ezpz launch --np 384 -ppn 12 --auto-retry --max-failover-retries 3 \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset metamathqa \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
    --resume_from_checkpoint "${RESUME_FROM}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 8 \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
