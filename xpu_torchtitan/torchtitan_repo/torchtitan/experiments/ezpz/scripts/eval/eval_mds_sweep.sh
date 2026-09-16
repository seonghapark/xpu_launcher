#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-mds-sweep
#PBS -j oe
#
# Convert + evaluate the AuroraGPT-2B Megatron-DeepSpeed checkpoints
# from the optimizer-experiments directory at 5K-step intervals.
#
# Usage:
#   qsub torchtitan/experiments/ezpz/scripts/eval/eval_mds_sweep.sh
#
# Override which stages/steps to evaluate via env vars:
#   STAGES="ntok4673B ntok7064B ntok7770B"   (default)
#   STEPS="5000 10000 ... 140000"            (default: 5K to 140K in 5K steps)

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1 2>/dev/null
export HF_HUB_ENABLE_HF_TRANSFER=0

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
source venvs/aurora/tt-lm-eval/bin/activate

CKPT_PFX="/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/Megatron-DeepSpeed/checkpoints"
STAGE_PFX="AuroraGPT-2B-ws3072-ds-stage0-nl12-hs2048-mb1-seq8192-gb6144-sp1-pp1-tp1-bf16-optsophiag-lr2.17e-5-lwf0.05"
STAGE_SFX="tokHF_tmgoogle_gemma-7b_flash"

STAGES="${STAGES:-ntok4673B ntok7064B ntok7770B}"

# Default: 5K to 140K in 5K steps. Add the very last step (140352) too.
if [[ -z "${STEPS:-}" ]]; then
    STEPS=""
    for s in $(seq 5000 5000 140000); do
        STEPS+=" $s"
    done
fi

OUT_BASE="outputs/evals/agpt-2b-mds"

for stage in ${STAGES}; do
    SRC_DIR="${CKPT_PFX}/${STAGE_PFX}_${stage}_${STAGE_SFX}"
    if [[ ! -d "${SRC_DIR}" ]]; then
        echo "[SKIP STAGE] ${stage}: ${SRC_DIR} not found"
        continue
    fi

    echo ""
    echo "##############################################"
    echo "# Stage: ${stage}"
    echo "# Source: ${SRC_DIR}"
    echo "##############################################"

    for step in ${STEPS}; do
        SRC="${SRC_DIR}/global_step${step}/mp_rank_00_model_states.pt"
        HF_DIR="${OUT_BASE}/${stage}/step-${step}/hf"
        RES_DIR="${OUT_BASE}/${stage}/step-${step}/results"

        if [[ ! -f "${SRC}" ]]; then
            echo "[SKIP] ${stage} step-${step}: ${SRC} not found"
            continue
        fi

        # Skip if already converted + evaluated
        if [[ -f "${RES_DIR}/results.json" ]]; then
            echo "[SKIP] ${stage} step-${step}: results.json exists"
            continue
        fi

        echo ""
        echo "======== ${stage} @ global_step${step} ========"

        # 1. Convert MDS -> HF (if not already)
        if [[ ! -f "${HF_DIR}/model-00001-of-00001.safetensors" ]]; then
            echo "Converting MDS -> HF..."
            python3 torchtitan/experiments/ezpz/eval/mds_to_hf.py \
                --mds_checkpoint "${SRC}" \
                --output_dir "${HF_DIR}" 2>&1
            if [[ $? -ne 0 ]]; then
                echo "[ERROR] conversion failed for step ${step}"
                continue
            fi
        else
            echo "HF checkpoint exists, skipping conversion."
        fi

        # 2. Run lm-eval (with the same monkey-patch we use for DCP evals)
        echo "Running lm-eval..."
        mkdir -p "${RES_DIR}"
        python3 << PYEOF
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None

import json
from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=${HF_DIR}",
    tasks=["hellaswag", "arc_easy", "arc_challenge", "winogrande"],
    batch_size=8,
    num_fewshot=0,
    device="xpu:0",
)
with open("${RES_DIR}/results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
for task, m in results["results"].items():
    acc = m.get("acc_norm,none") or m.get("acc,none", "?")
    print(f"  {task}: {acc:.4f}")
print(f"Done ${stage} step ${step}.")
PYEOF
    done
done

echo ""
echo "=== Sweep complete ==="
