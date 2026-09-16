#!/bin/bash --login
#
# GRPO smoke comparing the SFT'd checkpoint vs the raw pretrained
# baseline. The original justification for the 32N SFT push was
# "GRPO needs a stronger initialization." This validates whether the
# SFT'd model actually converges faster on a simple task.
#
# Runs 50 GRPO steps on `sum_digits` per model, back-to-back on the
# same idle allocation. We compare reward trajectory + completion
# quality.
#
# Usage (interactive, from a compute node with an existing allocation):
#   bash torchtitan/experiments/ezpz/rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh

set -o pipefail

SUBMIT_DIR="${PBS_O_WORKDIR:-/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan}"
cd "${SUBMIT_DIR}"

JOBID="${PBS_JOBID:-12468401.sunspot-pbs-0001.head.cm.sunspot.alcf.anl.gov}"
HOSTFILE="/var/spool/pbs/aux/${JOBID}"
if [[ ! -f "${HOSTFILE}" ]]; then
    echo "FATAL: hostfile ${HOSTFILE} doesn't exist"
    exit 1
fi
# Use just 4 nodes for the smoke (matches grpo_aurora2b.sh sizing).
# If a pre-staged hostfile already exists (set up by a wrapper for
# parallel-with-other-eval cases), respect it; otherwise pull the
# first 4 nodes from the PBS allocation hostfile.
if [[ ! -s /tmp/grpo-smoke-hostfile.txt ]]; then
    head -4 "${HOSTFILE}" > /tmp/grpo-smoke-hostfile.txt
fi
NHOSTS=$(wc -l < /tmp/grpo-smoke-hostfile.txt)
NRANKS=$(( NHOSTS * 12 ))

source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile 2>/dev/null
module load oneapi/release/2025.3.1 hdf5 pti-gpu 2>&1 | tail -1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

LOG_DIR="logs/grpo-smoke-sft-vs-baseline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${LOG_DIR}"

run_grpo() {
    local model_path="$1"
    local label="$2"
    local out_dir="outputs/grpo-smoke/${label}"
    mkdir -p "${out_dir}"
    echo ""
    echo "========================================================"
    echo "=== GRPO smoke: ${label}"
    echo "===              ${model_path}"
    echo "========================================================" | tee "${LOG_DIR}/${label}.log"
    time mpiexec \
        --envall --line-buffer \
        --np ${NRANKS} --ppn 12 \
        --hostfile /tmp/grpo-smoke-hostfile.txt \
        --cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 \
        python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
            --task sum_digits \
            --model_name_or_path "${model_path}" \
            --output_dir "${out_dir}" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --bf16 --beta 0.0 --fsdp full_shard \
            --max_steps 50 \
            --logging_steps 1 \
            --save_strategy no \
            --report_to wandb \
            2>&1 | tee -a "${LOG_DIR}/${label}.log" || true
    echo "=== ${label} done at $(date) ==="
}

run_grpo AuroraGPT-2B-sophiag-gs138650 baseline
run_grpo outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf sft-step729

echo ""
echo "=== Reward trajectory comparison ==="
echo "[baseline] last 5 reward log lines:"
grep -E "'reward':" "${LOG_DIR}/baseline.log" 2>/dev/null | tail -5
echo ""
echo "[sft-step729] last 5 reward log lines:"
grep -E "'reward':" "${LOG_DIR}/sft-step729.log" 2>/dev/null | tail -5
echo ""
echo "=== Logs in ${LOG_DIR}/ ==="
