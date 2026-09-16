#!/bin/bash
# convert_and_eval.sh — Convert DCP checkpoint to HF and run lm-eval
#
# Usage:
#   bash convert_and_eval.sh --model 2b --step 5000
#   bash convert_and_eval.sh --model 20b --step 1000 --tasks "hellaswag,mmlu"
#   bash convert_and_eval.sh --model 2b --step 5000 --convert-only
#   bash convert_and_eval.sh --model 2b --step 5000 --eval-only

set -euo pipefail

# ---- Defaults ----
MODEL=""
STEP=""
TASKS="${TASKS:-hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq}"
CONVERT_ONLY=false
EVAL_ONLY=false
EXPORT_DTYPE="${EXPORT_DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
# `--ckpt-name` overrides the auto-derived "agpt-${MODEL}-sophiag-olmo-mix-1124-n256-gbs3072"
# checkpoint name. Use this for v2 runs (e.g. n256-gbs6144 or n512-gbs12288).
CKPT_NAME_OVERRIDE=""
# `--repo-root` overrides the path to the clone whose outputs/ holds
# the DCP checkpoints. Defaults to this script's enclosing repo.
REPO_ROOT_OVERRIDE=""
# `--label` is appended to the output dir so v1 vs v2 results don't collide.
EVAL_LABEL=""

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --step) STEP="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --convert-only) CONVERT_ONLY=true; shift ;;
        --eval-only) EVAL_ONLY=true; shift ;;
        --export-dtype) EXPORT_DTYPE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
        --ckpt-name) CKPT_NAME_OVERRIDE="$2"; shift 2 ;;
        --repo-root) REPO_ROOT_OVERRIDE="$2"; shift 2 ;;
        --label) EVAL_LABEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" || -z "$STEP" ]]; then
    echo "Usage: $0 --model {2b|20b} --step STEP [--tasks TASKS] [--convert-only] [--eval-only]"
    exit 1
fi

# ---- Paths ----
SCRIPT_REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
REPO_ROOT="${REPO_ROOT_OVERRIDE:-$SCRIPT_REPO_ROOT}"
EVAL_DIR="${SCRIPT_REPO_ROOT}/torchtitan/experiments/ezpz/eval"
CKPT_BASE="${REPO_ROOT}/outputs/checkpoints"
TOKENIZER_DIR="${SCRIPT_REPO_ROOT}/assets/hf/gemma-7b"

# Checkpoint input (DCP format). Default name is the v1 layout
# (n256-gbs3072) — v2 callers MUST pass --ckpt-name.
CKPT_NAME="${CKPT_NAME_OVERRIDE:-agpt-${MODEL}-sophiag-olmo-mix-1124-n256-gbs3072}"
DCP_DIR="${CKPT_BASE}/${CKPT_NAME}/step-${STEP}"

# Output dirs. The label suffix prevents v1/v2 result collisions when
# they share a step number (e.g. both have a step-100).
LABEL_SUFFIX="${EVAL_LABEL:+-${EVAL_LABEL}}"
HF_DIR="${SCRIPT_REPO_ROOT}/outputs/evals/agpt-${MODEL}${LABEL_SUFFIX}/step-${STEP}/hf"
RESULTS_DIR="${SCRIPT_REPO_ROOT}/outputs/evals/agpt-${MODEL}${LABEL_SUFFIX}/step-${STEP}/results"

# HF config for this model size
HF_CONFIG="${EVAL_DIR}/configs/agpt_${MODEL}_config.json"

# Model flavor — config_registry uses lowercase "2b", "20b", etc.
MODEL_FLAVOR="${MODEL}"

# TP for vllm (2B fits on 1 GPU, 20B needs multiple)
if [[ "$MODEL" == "2b" ]]; then
    TP=1
elif [[ "$MODEL" == "20b" ]]; then
    TP=12
else
    TP=12
fi

echo "============================================"
echo "AGPT Evaluation Pipeline"
echo "============================================"
echo "Model:        agpt_${MODEL}"
echo "Step:         ${STEP}"
echo "DCP dir:      ${DCP_DIR}"
echo "HF output:    ${HF_DIR}"
echo "Results:      ${RESULTS_DIR}"
echo "Tasks:        ${TASKS}"
echo "TP:           ${TP}"
echo "Export dtype:  ${EXPORT_DTYPE}"
echo "============================================"

# ---- Validate inputs ----
if [[ ! -d "$DCP_DIR" ]]; then
    echo "ERROR: DCP checkpoint not found: ${DCP_DIR}"
    echo "Available steps:"
    ls "${CKPT_BASE}/${CKPT_NAME}/" 2>/dev/null | grep "step-" | sort -t- -k2 -n | tail -10
    exit 1
fi

if [[ ! -f "$HF_CONFIG" ]]; then
    echo "ERROR: HF config not found: ${HF_CONFIG}"
    exit 1
fi

if [[ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]]; then
    echo "ERROR: Tokenizer not found: ${TOKENIZER_DIR}"
    exit 1
fi

# ---- Step 1: Convert DCP → HuggingFace ----
if [[ "$EVAL_ONLY" != true ]]; then
    echo ""
    echo "[1/3] Converting DCP checkpoint to HuggingFace format..."
    mkdir -p "${HF_DIR}"

    python3 "${EVAL_DIR}/convert_to_hf.py" \
        "${DCP_DIR}" \
        "${HF_DIR}" \
        --model_name "experiments.ezpz.agpt" \
        --model_flavor "${MODEL_FLAVOR}" \
        --export_dtype "${EXPORT_DTYPE}"

    echo "[1/3] Conversion complete."

    # ---- Step 2: Copy config + tokenizer into HF dir ----
    echo ""
    echo "[2/3] Copying config.json and tokenizer files..."
    cp "${HF_CONFIG}" "${HF_DIR}/config.json"
    cp "${TOKENIZER_DIR}/tokenizer.json" "${HF_DIR}/"
    cp "${TOKENIZER_DIR}/tokenizer.model" "${HF_DIR}/"
    cp "${TOKENIZER_DIR}/tokenizer_config.json" "${HF_DIR}/"
    cp "${TOKENIZER_DIR}/special_tokens_map.json" "${HF_DIR}/"

    echo "[2/3] Assets copied."
else
    echo ""
    echo "[1-2/3] Skipping conversion (--eval-only)"
    if [[ ! -d "$HF_DIR" ]]; then
        echo "ERROR: HF checkpoint not found at ${HF_DIR}. Run without --eval-only first."
        exit 1
    fi
fi

# ---- Step 3: Run lm-eval ----
if [[ "$CONVERT_ONLY" != true ]]; then
    echo ""
    echo "[3/3] Running lm-eval with vllm backend..."
    mkdir -p "${RESULTS_DIR}"

    lm_eval \
        --model vllm \
        --model_args "pretrained=${HF_DIR},tensor_parallel_size=${TP},dtype=auto,gpu_memory_utilization=0.8,max_model_len=4096" \
        --tasks "${TASKS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_fewshot "${NUM_FEWSHOT}" \
        --output_path "${RESULTS_DIR}" \
        --log_samples

    echo "[3/3] Evaluation complete."
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo "View with: cat ${RESULTS_DIR}/results.json | python3 -m json.tool"
else
    echo ""
    echo "[3/3] Skipping evaluation (--convert-only)"
fi

echo ""
echo "Done."
