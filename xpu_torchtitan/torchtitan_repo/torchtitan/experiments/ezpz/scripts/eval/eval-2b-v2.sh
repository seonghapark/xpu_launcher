#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -N eval-2b-v2
#PBS -j oe
#
# Convert + eval the v2 2B SophiaG checkpoints (fp32 master) and
# produce results that can be directly compared against the v1 256N
# eval table in docs/evals/agpt/2b/README.md (and v1 256N+512N eval
# results sit at outputs/evals/agpt-2b/).
#
# v2 ckpt paths (both 256N and 512N v2 runs):
#   /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/
#     outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-{N}
#     outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288/step-{N}
#
# Default: eval the 256N v2 checkpoint at every 200 steps from 200 to 2000
# (10 ckpts). Override via STEPS / CKPT_NAME env vars.
#
# Output (in this clone):
#   outputs/evals/agpt-2b-v2/step-{N}/{hf,results}/

# PBS scripts must NOT use `set -euo pipefail` per CLAUDE.md.
set -o pipefail

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HF_HUB_ENABLE_HF_TRANSFER=0

module load oneapi/release/2025.3.1 hdf5 pti-gpu frameworks/2025.3.1
echo "PWD: $(pwd)"
echo "Modules loaded."

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

# V2_REPO controls where the DCP checkpoints LIVE (env-overridable so we
# can also eval ckpts in the legacy /flare/.../projects/saforem2/torchtitan/
# clone, where the chain pre-2026-04-30 was written).
V2_REPO="${V2_REPO:-/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz}"
# USE_LEGACY_CONVERTER=1 -> call convert_to_hf_legacy.py (handles the
# pre-qkv_linear FQN layout used by ckpts in the legacy clone). Default 0
# uses the canonical convert_to_hf.py.
USE_LEGACY_CONVERTER="${USE_LEGACY_CONVERTER:-0}"
# CONVERT_REPO controls where the conversion ENV + script live (the .venv
# and torchtitan/experiments/ezpz/eval/convert_to_hf.py). This MUST be a
# clone that has the eval/ subdir + working torch 2.13 venv — i.e. the
# canonical v2 clone. Default to PWD (where this script is executed from)
# so it works without override; only override if you have a newer eval env
# in a different clone.
CONVERT_REPO="${CONVERT_REPO:-/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz}"
# Default to the canonical 512N chain (gbs12288); override CKPT_NAME +
# LABEL to evaluate other trajectories (e.g. the abandoned 256N
# one-shot).
V2_CKPT_NAME="${CKPT_NAME:-agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288}"
# LABEL is appended to the output dir so 256N + 512N evals can
# coexist under outputs/evals/agpt-2b-v2-<LABEL>/.
LABEL="${LABEL:-512n}"

STEPS="${STEPS:-1000 2000 3000 4000 5000}"
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"

for step in $STEPS; do
    DCP_DIR="${V2_REPO}/outputs/checkpoints/${V2_CKPT_NAME}/step-${step}"
    HF_DIR="outputs/evals/agpt-2b-v2-${LABEL}/step-${step}/hf"
    RESULTS_DIR="outputs/evals/agpt-2b-v2-${LABEL}/step-${step}/results"

    if [[ ! -d "$DCP_DIR" ]]; then
        echo "[SKIP] 2b-v2 step-${step}: no DCP at ${DCP_DIR}"
        continue
    fi
    if [[ -f "${RESULTS_DIR}/results.json" ]]; then
        echo "[SKIP] 2b-v2 step-${step}: results.json already exists"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "2B v2 step-${step}"
    echo "============================================================"

    EVAL_CLONE="$(pwd)"
    HF_DIR_ABS="${EVAL_CLONE}/${HF_DIR}"
    RESULTS_DIR_ABS="${EVAL_CLONE}/${RESULTS_DIR}"

    # ---- Step 1: DCP -> HF (run from CONVERT_REPO so torchtitan resolves) ----
    # NOTE: V2_REPO can point at any clone (e.g. legacy) so DCP can be sourced
    # from there, but CONVERT_REPO MUST be the canonical v2 clone with .venv +
    # torchtitan/experiments/ezpz/eval/convert_to_hf.py. Don't conflate them.
    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] Converting DCP -> HF (venv from ${CONVERT_REPO}, DCP from ${V2_REPO})..."
        # Validate CONVERT_REPO has what we need before launching the subshell.
        if [[ ! -f "${CONVERT_REPO}/.venv/bin/activate" ]]; then
            echo "  ERROR: ${CONVERT_REPO}/.venv/bin/activate missing — set CONVERT_REPO to a clone with a built .venv"
            echo "[1/2] Conversion FAILED — skipping eval for step ${step}"
            continue
        fi
        if [[ "${USE_LEGACY_CONVERTER}" == "1" ]]; then
            CONVERT_PY="torchtitan/experiments/ezpz/eval/convert_to_hf_legacy.py"
        else
            CONVERT_PY="torchtitan/experiments/ezpz/eval/convert_to_hf.py"
        fi
        if [[ ! -f "${CONVERT_REPO}/${CONVERT_PY}" ]]; then
            echo "  ERROR: ${CONVERT_REPO}/${CONVERT_PY} missing"
            echo "[1/2] Conversion FAILED — skipping eval for step ${step}"
            continue
        fi
        mkdir -p "${HF_DIR_ABS}"
        (
            cd "${CONVERT_REPO}" || exit 1
            source .venv/bin/activate
            echo "  subshell: pwd=$(pwd)"
            echo "  subshell: which python3=$(which python3)"
            echo "  subshell: converter=${CONVERT_PY}"
            PYTHONPATH=".:${PYTHONPATH:-}" python3 -c "import torchtitan; print('  torchtitan from:', torchtitan.__file__)" \
                || { echo "  ERROR: torchtitan import failed"; exit 1; }
            PYTHONPATH=".:${PYTHONPATH:-}" python3 "${CONVERT_PY}" \
                "${DCP_DIR}" \
                "${HF_DIR_ABS}" \
                --model_name "experiments.ezpz.agpt" \
                --model_flavor "2b" \
                --export_dtype "bfloat16"
        ) || { echo "[1/2] Conversion FAILED — skipping eval for step ${step}"; continue; }
        cp "${EVAL_CLONE}/torchtitan/experiments/ezpz/eval/configs/agpt_2b_config.json" \
            "${HF_DIR_ABS}/config.json"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer.{json,model} "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/tokenizer_config.json "${HF_DIR_ABS}/"
        cp "${EVAL_CLONE}"/assets/hf/gemma-7b/special_tokens_map.json "${HF_DIR_ABS}/"
        echo "[1/2] Conversion done."
    else
        echo "[1/2] HF already converted, skipping."
    fi

    if [[ ! -f "${HF_DIR_ABS}/model.safetensors.index.json" \
          && ! -f "${HF_DIR_ABS}/model.safetensors" ]]; then
        echo "[1/2] No model file in ${HF_DIR_ABS} — skipping eval"
        continue
    fi

    # ---- Step 2: lm-eval (frameworks venv + tt-lm-eval overlay) ----
    echo "[2/2] Running lm-eval..."
    source venvs/aurora/tt-lm-eval/bin/activate
    mkdir -p "${RESULTS_DIR_ABS}"
    # batch_size=8 is fine for 2B on single XPU (vs 2 for 20B).
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
    batch_size=8,
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
echo "=== 2B v2 eval complete ==="
