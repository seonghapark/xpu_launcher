#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Benchmark/smoke-test script for ezpz experiment configs.
# Runs a few training iterations for each config, parses metrics, and
# generates a summary report with timing/throughput numbers.
#
# Usage:
#   BENCH_STEPS=5 bash torchtitan/experiments/ezpz/scripts/run_benchmarks.sh
#
# Run only specific configs:
#   BENCH_CONFIGS="agpt_2b agpt_20b moe_2b" bash torchtitan/experiments/ezpz/scripts/run_benchmarks.sh
#
# Per-config TP override (config:tp format):
#   BENCH_CONFIGS="agpt_2b agpt_80b:2 moe_7b" bash torchtitan/experiments/ezpz/scripts/run_benchmarks.sh
#
# Extra CLI args are forwarded to every run:
#   bash torchtitan/experiments/ezpz/scripts/run_benchmarks.sh --parallelism.tp_degree 2
#
# Environment variables:
#   BENCH_CONFIGS   — space-separated configs to run (default: all agpt + moe)
#                     Format: "config_name" or "config_name:tp_degree"
#                     Module is inferred: agpt_* -> ezpz.agpt, moe_* -> ezpz.moe
#   BENCH_STEPS     — training iterations per run (default: 10)
#   BENCH_SEQ_LEN   — sequence length (default: 8192)
#   BENCH_LOCAL_BS  — local batch size (default: 1)
#   BENCH_GAS       — gradient accumulation steps (default: 1)
#   BENCH_TP        — default tensor parallel degree (default: 1)
#   BENCH_PP        — pipeline parallel degree (default: 1)
#   BENCH_TIMEOUT   — per-run timeout in seconds (default: 1800)
#   FILTER_NONZERO_RANKS — set to 1 to suppress output from non-rank-0 (default: 0)
#   NO_COMPILE      — set to 1 to disable torch.compile (default: 0)
#   BENCH_OUTDIR    — override output directory (default: outputs/benchmarks/TIMESTAMP)

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup (set +u needed: lmod/ezpz reference unset vars)
# ---------------------------------------------------------------------------
set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi
set -u

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENCH_STEPS="${BENCH_STEPS:-10}"
BENCH_SEQ_LEN="${BENCH_SEQ_LEN:-8192}"
BENCH_LOCAL_BS="${BENCH_LOCAL_BS:-1}"
BENCH_GAS="${BENCH_GAS:-1}"
BENCH_TP="${BENCH_TP:-1}"
BENCH_PP="${BENCH_PP:-1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
FILTER_NONZERO_RANKS="${FILTER_NONZERO_RANKS:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${BENCH_OUTDIR:-outputs/benchmarks/${TIMESTAMP}}"
mkdir -p "${OUTDIR}"

# Default: all registered agpt + moe configs
DEFAULT_CONFIGS="agpt_debugmodel agpt_2b agpt_7b agpt_8b agpt_20b agpt_50b"
DEFAULT_CONFIGS+=" agpt_80b:2 agpt_80b_alt:2 agpt_80b_wide:2 agpt_80b_deep:2 agpt_80b_deep_alt:2"
DEFAULT_CONFIGS+=" moe_debugmodel moe_500m moe_2b moe_4b moe_7b moe_10b_2b moe_10b_2b_sdpa"

BENCH_CONFIGS="${BENCH_CONFIGS:-${DEFAULT_CONFIGS}}"
read -ra CONFIG_SPECS <<< "${BENCH_CONFIGS}"

# ---------------------------------------------------------------------------
# Parse config specs into parallel arrays
# ---------------------------------------------------------------------------
declare -a LABELS MODULES CONFIGS TPS_OVERRIDES

for ((i = 0; i < ${#CONFIG_SPECS[@]}; i++)); do
    spec="${CONFIG_SPECS[$i]}"

    # Parse config:tp format
    if [[ "${spec}" == *:* ]]; then
        config="${spec%%:*}"
        tp_override="${spec##*:}"
    else
        config="${spec}"
        tp_override=0
    fi

    # Infer module from config prefix
    if [[ "${config}" == agpt_* ]]; then
        module="ezpz.agpt"
    elif [[ "${config}" == moe_* ]]; then
        module="ezpz.moe"
    else
        echo "WARNING: unknown config prefix for '${config}', assuming ezpz.agpt"
        module="ezpz.agpt"
    fi

    LABELS[$i]="${config}"
    MODULES[$i]="${module}"
    CONFIGS[$i]="${config}"
    TPS_OVERRIDES[$i]="${tp_override}"
done

NUM_CONFIGS="${#LABELS[@]}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Kill stale python processes from previous runs
# ---------------------------------------------------------------------------
echo "--- Cleaning up stale processes and cache ---"
pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true
# rm -rf .cache/blendcorpus/*.npy 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
RUN_DATE="$(date -Iseconds)"
MACHINE_NAME="$(ezpz_get_machine_name 2>/dev/null || hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
DEVICES_PER_NODE=$(("${NGPUS}" / NUM_NODES))

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
declare -a WALL_TIMES STATUSES TPS_VALUES TFLOPS_VALUES MFU_VALUES MEMORY_VALUES WANDB_VALUES

echo "============================================================"
echo " ezpz benchmarks — ${TIMESTAMP}"
echo " steps=${BENCH_STEPS}  devices=${NGPUS}  nodes=${NUM_NODES}"
echo " configs: ${BENCH_CONFIGS}"
echo "============================================================"
echo ""

for ((i = 0; i < NUM_CONFIGS; i++)); do
    label="${LABELS[$i]}"
    module="${MODULES[$i]}"
    config="${CONFIGS[$i]}"
    logfile="${OUTDIR}/${label}.log"

    echo "--- [${label}] running (module=${module} config=${config}) ---"
    echo "    started @ $(date +%Y%m%d-%H%M%S)"
    echo "    logfile: ${logfile}"
    echo ""

    start_seconds=$SECONDS

    compile_args=()
    if ((NO_COMPILE)); then
        compile_args=("--compile.no-enable")
    fi

    # Per-config TP (0 means use BENCH_TP default)
    tp="${TPS_OVERRIDES[$i]}"
    if ((tp == 0)); then tp="${BENCH_TP}"; fi

    # Compute global batch size to enable gradient accumulation:
    # GBS = DP * local_batch_size * GAS, where DP = NGPUS / TP / PP
    global_bs=$((NGPUS * BENCH_LOCAL_BS * BENCH_GAS / tp / BENCH_PP))

    timeout "${BENCH_TIMEOUT}" \
        stdbuf -oL -eL \
        env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module "${module}" \
        --config "${config}" \
        --training.steps "${BENCH_STEPS}" \
        --training.local_batch_size "${BENCH_LOCAL_BS}" \
        --training.global_batch_size "${global_bs}" \
        --training.seq_len "${BENCH_SEQ_LEN}" \
        --parallelism.tensor_parallel_degree "${tp}" \
        --parallelism.pipeline_parallel_degree "${BENCH_PP}" \
        --metrics.log_freq 1 \
        --checkpoint.no-enable \
        --dataloader.dataset blendcorpus \
        --dataloader.dataset_path "${DATASET_PATH}" \
        "${compile_args[@]}" \
        "$@" \
        2>&1 | if ((FILTER_NONZERO_RANKS)); then grep -v '^\[rank[1-9][0-9]*\]:'; else cat; fi >"${logfile}" || true
    exit_code=${PIPESTATUS[0]}

    # Kill any leftover processes from this run
    pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null || true
    sleep 2

    # Determine run status
    if ((exit_code == 124)); then
        STATUSES[$i]="TIMEOUT"
    elif ((exit_code != 0)); then
        STATUSES[$i]="FAIL(rc=${exit_code})"
    elif grep -q 'OUT_OF_RESOURCES\|out of memory\|OOM' "${logfile}"; then
        STATUSES[$i]="OOM"
    elif grep -q 'Traceback\|Error\|Exception' "${logfile}" && ! grep -q 'loss:' "${logfile}"; then
        STATUSES[$i]="CRASH"
    elif ! grep -q 'loss:' "${logfile}"; then
        STATUSES[$i]="NO_OUTPUT"
    else
        STATUSES[$i]="OK"
    fi

    elapsed=$((SECONDS - start_seconds))
    WALL_TIMES[$i]="${elapsed}"

    # Parse metrics from last log line containing "loss:"
    last_line="$(grep 'loss:' "${logfile}" | tail -1 || true)"

    if [[ -n "${last_line}" ]]; then
        TPS_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*tps: \([0-9,]*\).*/\1/p' | tr -d ',')"
        TFLOPS_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*tflops: \([0-9,.]*\).*/\1/p' | tr -d ',')"
        MFU_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*mfu: \([0-9.]*\)%.*/\1/p')"
        MEMORY_VALUES[$i]="$(echo "${last_line}" | sed -n 's/.*memory: \([0-9.]*GiB([0-9.]*%)\).*/\1/p')"
    fi

    # Parse wandb run URL
    WANDB_VALUES[$i]="$(grep -m1 'View run at' "${logfile}" | sed -n 's/.*View run at \(https:[^ ]*\).*/\1/p' || true)"

    # Fill in defaults for missing values
    TPS_VALUES[$i]="${TPS_VALUES[$i]:-N/A}"
    TFLOPS_VALUES[$i]="${TFLOPS_VALUES[$i]:-N/A}"
    MFU_VALUES[$i]="${MFU_VALUES[$i]:-N/A}"
    MEMORY_VALUES[$i]="${MEMORY_VALUES[$i]:-N/A}"
    WANDB_VALUES[$i]="${WANDB_VALUES[$i]:-}"

    echo "    status=${STATUSES[$i]}  wall=${elapsed}s  memory=${MEMORY_VALUES[$i]}  tps=${TPS_VALUES[$i]}  tflops=${TFLOPS_VALUES[$i]}  mfu=${MFU_VALUES[$i]}"
    echo ""
done

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# ezpz Benchmark Report"
    echo ""

    # Metadata table
    _meta_keys=("Date" "Commit" "Machine" "Job ID" "Nodes" "Devices" "Devices/Node" "Steps")
    _meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPUS}" "${DEVICES_PER_NODE}" "${BENCH_STEPS}")
    _vw=5
    for _v in "${_meta_vals[@]}"; do
        ((${#_v} > _vw)) && _vw=${#_v}
    done

    printf "| %-12s | %-${_vw}s |\n" "Field" "Value"
    printf "|-%s-|-%s-|\n" "$(printf '%0.s-' $(seq 1 12))" "$(printf '%0.s-' $(seq 1 "${_vw}"))"
    for ((_j = 0; _j < ${#_meta_keys[@]}; _j++)); do
        printf "| %-12s | %-${_vw}s |\n" "${_meta_keys[$_j]}" "${_meta_vals[$_j]}"
    done
    echo ""

    echo "## Results"
    echo ""

    # Add wandb column only if any run has a URL
    has_wandb=false
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        if [[ -n "${WANDB_VALUES[$i]}" ]]; then
            has_wandb=true
            break
        fi
    done

    # Compute Config column width from longest label
    cw=6  # minimum ("Config")
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        ((${#LABELS[$i]} > cw)) && cw=${#LABELS[$i]}
    done

    # Build wandb shortlink column values: [run_id](url)
    declare -a WANDB_CELLS
    ww=3  # minimum ("W&B")
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        if [[ -n "${WANDB_VALUES[$i]}" ]]; then
            run_id="${WANDB_VALUES[$i]##*/}"
            WANDB_CELLS[$i]="[${run_id}](${WANDB_VALUES[$i]})"
        else
            WANDB_CELLS[$i]=""
        fi
        ((${#WANDB_CELLS[$i]} > ww)) && ww=${#WANDB_CELLS[$i]}
    done

    # Header
    printf "| %-${cw}s | %5s | %20s | %7s | %8s | %6s | %10s |" \
        "Config" "Steps" "Memory" "TPS" "TFLOPS" "MFU" "Wall (s)"
    if $has_wandb; then
        printf " %-${ww}s |" "W&B"
    fi
    printf " %-6s |\n" "Status"

    # Separator
    _sep() { printf '%0.s-' $(seq 1 "$1"); }
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|" \
        "$(_sep "${cw}")" "$(_sep 5)" "$(_sep 20)" \
        "$(_sep 7)" "$(_sep 8)" "$(_sep 6)" "$(_sep 10)"
    if $has_wandb; then
        printf -- "-%s-|" "$(_sep "${ww}")"
    fi
    printf -- "-%s-|\n" "$(_sep 6)"

    # Rows
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        printf "| %-${cw}s | %5s | %20s | %7s | %8s | %6s | %10s |" \
            "${LABELS[$i]}" \
            "${BENCH_STEPS}" \
            "${MEMORY_VALUES[$i]}" \
            "${TPS_VALUES[$i]}" \
            "${TFLOPS_VALUES[$i]}" \
            "${MFU_VALUES[$i]}" \
            "${WALL_TIMES[$i]}"
        if $has_wandb; then
            printf " %-${ww}s |" "${WANDB_CELLS[$i]}"
        fi
        printf " %-6s |\n" "${STATUSES[$i]}"
    done

    echo ""
    echo "Logs: \`${OUTDIR}/\`"
} >"${REPORT}"

# Print report to stdout
echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Logs saved to:   ${OUTDIR}/"
