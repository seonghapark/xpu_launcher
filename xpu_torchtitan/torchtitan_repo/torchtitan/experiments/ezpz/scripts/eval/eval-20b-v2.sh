#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-20b-v2
#PBS -j oe
#
# Convert + eval the v2 20B SophiaG checkpoints (fp32 master) and
# produce results that can be directly compared against the v1
# bf16-tainted eval table in docs/evals/agpt/20b/README.md.
#
# v2 ckpt path:
#   /flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/
#     outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288/step-{N}
#
# Output (in this clone, under outputs/evals/agpt-20b-v2/):
#   step-100/hf/      HF safetensors + tokenizer
#   step-100/results/ lm-eval JSON
#
# Steps to evaluate are passed via STEPS env var (space-separated). At
# write time only step-100 + step-200 are saved; later runs of this
# script will pick up newer ckpts as they land.

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md — venv
# activate has unbound vars and would trigger on first source.
set -o pipefail

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1
echo "PWD: $(pwd)"
echo "Modules loaded."

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# This eval pipeline runs against the bare frameworks/2025.3.1 module
# stack (NOT the user venv) per CLAUDE.md — user venv has transformers
# 5.6.2 which breaks lm-eval's HF backend.
#
# But the convert_to_hf step needs torchtitan + the v2 model registry —
# so source the v2 venv for the conversion, then deactivate before
# the lm-eval step.
V2_REPO="/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz"
# Default to the canonical 512N chain (gbs12288); override CKPT_NAME +
# LABEL to evaluate other trajectories (e.g. the 256N comparator).
V2_CKPT_NAME="${CKPT_NAME:-agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288}"
# LABEL is appended to the output dir so 256N + 512N evals can
# coexist under outputs/evals/agpt-20b-v2-<LABEL>/.
LABEL="${LABEL:-512n}"

STEPS="${STEPS:-100 200 300}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"

for step in $STEPS; do
    DCP_DIR="${V2_REPO}/outputs/checkpoints/${V2_CKPT_NAME}/step-${step}"
    HF_DIR="outputs/evals/agpt-20b-v2-${LABEL}/step-${step}/hf"
    RESULTS_DIR="outputs/evals/agpt-20b-v2-${LABEL}/step-${step}/results"

    if [[ ! -d "$DCP_DIR" ]]; then
        echo "[SKIP] 20b-v2 step-${step}: no DCP at ${DCP_DIR}"
        continue
    fi
    if [[ -f "${RESULTS_DIR}/results.json" ]]; then
        echo "[SKIP] 20b-v2 step-${step}: results.json already exists"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "20B v2 step-${step}"
    echo "============================================================"

    # Use absolute paths because step 1 (conversion) cd's into the v2
    # clone, then step 2 (lm-eval) cd's back here.
    EVAL_CLONE="$(pwd)"
    HF_DIR_ABS="${EVAL_CLONE}/${HF_DIR}"
    RESULTS_DIR_ABS="${EVAL_CLONE}/${RESULTS_DIR}"

    # ---- Step 1: DCP -> HF (run from v2 clone so torchtitan resolves) ----
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] Converting DCP -> HF (via v2 venv, from ${V2_REPO})..."
        mkdir -p "${HF_DIR_ABS}"
        # Run conversion from the v2 clone so its torchtitan + venv +
        # local model registry are all on PYTHONPATH. Use PYTHONPATH=.
        # to force-add the v2 clone's torchtitan/ ahead of any system
        # python tree.
        (
            cd "${V2_REPO}" || exit 1
            source .venv/bin/activate
            echo "  subshell: pwd=$(pwd)"
            echo "  subshell: which python3=$(which python3)"
            echo "  subshell: torchtitan check..."
            PYTHONPATH=".:${PYTHONPATH:-}" python3 -c "import torchtitan; print('  torchtitan from:', torchtitan.__file__)" \
                || { echo "  ERROR: torchtitan import failed"; exit 1; }
            PYTHONPATH=".:${PYTHONPATH:-}" python3 torchtitan/experiments/ezpz/eval/convert_to_hf.py \
                "${DCP_DIR}" \
                "${HF_DIR_ABS}" \
                --model_name "experiments.ezpz.agpt" \
                --model_flavor "20b" \
                --export_dtype "bfloat16"
        ) || { echo "[1/2] Conversion FAILED — skipping eval for step ${step}"; continue; }
        # Copy HF config + tokenizer assets from the eval clone (we
        # already have these, no need to pull from v2 clone).
        cp "${EVAL_CLONE}/torchtitan/experiments/ezpz/eval/configs/agpt_20b_config.json" \
            "${HF_DIR_ABS}/config.json"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer.{json,model} "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer_config.json "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/special_tokens_map.json "${HF_DIR_ABS}/"
        echo "[1/2] Conversion done."
    else
        echo "[1/2] HF already converted, skipping."
    fi

    # Sanity check: bail if conversion produced no model file.
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] No model file in ${HF_DIR_ABS} — skipping eval"
        continue
    fi

    # ---- Step 2: lm-eval (bare frameworks venv + tt-lm-eval overlay) ----
    echo "[2/2] Running lm-eval..."
    source venvs/aurora/tt-lm-eval/bin/activate
    mkdir -p "${RESULTS_DIR_ABS}"
    HF_DIR_ABS="${HF_DIR_ABS}" RESULTS_DIR_ABS="${RESULTS_DIR_ABS}" TASKS="${TASKS}" \
    python3 << 'PYEOF'
import os, json
import transformers.modeling_utils as mu
mu.caching_allocator_warmup = lambda *args, **kwargs: None
from lm_eval import evaluator

hf_dir = os.environ["HF_DIR_ABS"]
results_dir = os.environ["RESULTS_DIR_ABS"]
tasks = os.environ["TASKS"].split(",")

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={hf_dir}",
    tasks=tasks,
    batch_size=2,
    num_fewshot=0,
    device="xpu:0",
)
with open(f"{results_dir}/results.json", "w") as f:
    json.dump(results["results"], f, indent=2)
for task, metrics in results["results"].items():
    acc = metrics.get("acc_norm,none") or metrics.get("acc,none", "?")
    print(f"  {task}: {acc:.4f}" if isinstance(acc, float) else f"  {task}: {acc}")
PYEOF
    deactivate
    echo "[2/2] Eval done. results: ${RESULTS_DIR_ABS}/results.json"
done

echo ""
echo "=== 20B v2 eval complete ==="
