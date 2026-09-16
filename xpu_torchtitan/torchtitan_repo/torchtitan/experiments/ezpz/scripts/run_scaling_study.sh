#!/usr/bin/bash
# Scaling study benchmark for ezpz experiment configs.
# Runs 5 models for BENCH_STEPS iterations each, parses metrics,
# and generates report.md + results.json for aggregation.
#
# Usage (inside a PBS allocation):
#   bash torchtitan/experiments/ezpz/scripts/run_scaling_study.sh
#
# Extra CLI args are forwarded to every run:
#   bash torchtitan/experiments/ezpz/scripts/run_scaling_study.sh --debug.deterministic
#
# Environment variables:
#   BENCH_STEPS     — training iterations per run (default: 20)
#   BENCH_TIMEOUT   — per-run timeout in seconds (default: 2400)
#   FILTER_NONZERO_RANKS — set to 1 to suppress non-rank-0 output (default: 0)
#   NO_COMPILE      — set to 1 to disable torch.compile (default: 0)
#   SCALING_OUTDIR  — override output directory (default: outputs/scaling_study/TIMESTAMP)

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
BENCH_STEPS="${BENCH_STEPS:-20}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-2400}"
FILTER_NONZERO_RANKS="${FILTER_NONZERO_RANKS:-0}"
NO_COMPILE="${NO_COMPILE:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
DEVICES_PER_NODE=$(( NGPUS / NUM_NODES ))

OUTDIR="${SCALING_OUTDIR:-outputs/scaling_study/${TIMESTAMP}/n${NUM_NODES}}"
mkdir -p "${OUTDIR}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Model configurations (parallel arrays)
# ---------------------------------------------------------------------------
#              agpt_2b     agpt_20b    agpt_80b    moe_2b      moe_7b
LABELS=(       "agpt_2b"   "agpt_20b"  "agpt_80b"  "moe_2b"    "moe_7b"    )
MODULES=(      "ezpz.agpt" "ezpz.agpt" "ezpz.agpt" "ezpz.moe"  "ezpz.moe"  )
CONFIGS=(      "agpt_2b"   "agpt_20b"  "agpt_80b"  "moe_2b"    "moe_7b"    )
LBS_VALS=(     1           1           1           16          2           )
TP_VALS=(      1           1           2           1           1           )
GAS_VALS=(     1           2           1           1           1           )
SEQ_VALS=(     8192        8192        8192        4096        4096        )

NUM_CONFIGS="${#LABELS[@]}"

# ---------------------------------------------------------------------------
# Kill stale python processes from previous runs
# ---------------------------------------------------------------------------
echo "--- Cleaning up stale processes and cache ---"
pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true
rm -rf .cache/blendcorpus/*.npy 2>/dev/null || true
echo ""

# ---------------------------------------------------------------------------
# Job metadata
# ---------------------------------------------------------------------------
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
RUN_DATE="$(date -Iseconds)"
MACHINE_NAME="$(ezpz_get_machine_name 2>/dev/null || hostname -s)"
JOB_ID="${PBS_JOBID:-${SLURM_JOB_ID:-${COBALT_JOBID:-local}}}"

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------
declare -a WALL_TIMES STATUSES TPS_VALUES TFLOPS_VALUES MFU_VALUES MEMORY_VALUES WANDB_VALUES

echo "============================================================"
echo " Scaling Study — ${TIMESTAMP}"
echo " steps=${BENCH_STEPS}  devices=${NGPUS}  nodes=${NUM_NODES}"
echo " models: ${LABELS[*]}"
echo "============================================================"
echo ""

for ((i = 0; i < NUM_CONFIGS; i++)); do
    label="${LABELS[$i]}"
    module="${MODULES[$i]}"
    config="${CONFIGS[$i]}"
    lbs="${LBS_VALS[$i]}"
    tp="${TP_VALS[$i]}"
    gas="${GAS_VALS[$i]}"
    seq_len="${SEQ_VALS[$i]}"
    logfile="${OUTDIR}/${label}.log"

    # Compute GBS: DP = NGPUS / TP, GBS = DP * LBS * GAS
    dp=$(( NGPUS / tp ))
    gbs=$(( dp * lbs * gas ))

    echo "--- [${label}] running (module=${module} config=${config}) ---"
    echo "    TP=${tp} DP=${dp} LBS=${lbs} GAS=${gas} GBS=${gbs} seq_len=${seq_len}"
    echo "    started @ $(date +%Y%m%d-%H%M%S)"
    echo "    logfile: ${logfile}"
    echo ""

    start_seconds=$SECONDS

    compile_args=()
    if ((NO_COMPILE)); then
        compile_args=("--compile.no-enable")
    fi

    timeout "${BENCH_TIMEOUT}" \
        stdbuf -oL -eL \
        env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module "${module}" \
        --config "${config}" \
        --training.steps "${BENCH_STEPS}" \
        --training.local_batch_size "${lbs}" \
        --training.global_batch_size "${gbs}" \
        --training.seq_len "${seq_len}" \
        --parallelism.tensor_parallel_degree "${tp}" \
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
# Generate markdown report
# ---------------------------------------------------------------------------
REPORT="${OUTDIR}/report.md"

{
    echo "# Scaling Study Report — ${NUM_NODES} Nodes"
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

    # Check for wandb
    has_wandb=false
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        if [[ -n "${WANDB_VALUES[$i]}" ]]; then
            has_wandb=true
            break
        fi
    done

    # Compute Config column width
    cw=8
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        ((${#LABELS[$i]} > cw)) && cw=${#LABELS[$i]}
    done

    # Header
    printf "| %-${cw}s | %3s | %3s | %5s | %5s | %20s | %9s | %8s | %6s | %10s |" \
        "Config" "TP" "LBS" "GAS" "GBS" "Memory" "TPS" "TFLOPS" "MFU" "Wall (s)"
    if $has_wandb; then printf " %-6s |" "W&B"; fi
    printf " %-6s |\n" "Status"

    # Separator
    _sep() { printf '%0.s-' $(seq 1 "$1"); }
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|" \
        "$(_sep "${cw}")" "$(_sep 3)" "$(_sep 3)" "$(_sep 5)" "$(_sep 5)" \
        "$(_sep 20)" "$(_sep 9)" "$(_sep 8)" "$(_sep 6)" "$(_sep 10)"
    if $has_wandb; then printf -- "-%s-|" "$(_sep 6)"; fi
    printf -- "-%s-|\n" "$(_sep 6)"

    # Rows
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        tp="${TP_VALS[$i]}"
        dp=$(( NGPUS / tp ))
        gbs=$(( dp * LBS_VALS[$i] * GAS_VALS[$i] ))
        printf "| %-${cw}s | %3s | %3s | %5s | %5s | %20s | %9s | %8s | %6s | %10s |" \
            "${LABELS[$i]}" \
            "${tp}" \
            "${LBS_VALS[$i]}" \
            "${GAS_VALS[$i]}" \
            "${gbs}" \
            "${MEMORY_VALUES[$i]}" \
            "${TPS_VALUES[$i]}" \
            "${TFLOPS_VALUES[$i]}" \
            "${MFU_VALUES[$i]}" \
            "${WALL_TIMES[$i]}"
        if $has_wandb; then
            if [[ -n "${WANDB_VALUES[$i]}" ]]; then
                run_id="${WANDB_VALUES[$i]##*/}"
                printf " [%s](%s) |" "${run_id}" "${WANDB_VALUES[$i]}"
            else
                printf " %-6s |" ""
            fi
        fi
        printf " %-6s |\n" "${STATUSES[$i]}"
    done

    echo ""
    echo "Logs: \`${OUTDIR}/\`"
} >"${REPORT}"

# ---------------------------------------------------------------------------
# Generate machine-readable results.json
# ---------------------------------------------------------------------------
RESULTS_JSON="${OUTDIR}/results.json"

{
    echo "{"
    echo "  \"timestamp\": \"${TIMESTAMP}\","
    echo "  \"date\": \"${RUN_DATE}\","
    echo "  \"machine\": \"${MACHINE_NAME}\","
    echo "  \"job_id\": \"${JOB_ID}\","
    echo "  \"commit\": \"${GIT_COMMIT}\","
    echo "  \"nodes\": ${NUM_NODES},"
    echo "  \"devices\": ${NGPUS},"
    echo "  \"devices_per_node\": ${DEVICES_PER_NODE},"
    echo "  \"steps\": ${BENCH_STEPS},"
    echo "  \"results\": ["
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        tp="${TP_VALS[$i]}"
        dp=$(( NGPUS / tp ))
        gbs=$(( dp * LBS_VALS[$i] * GAS_VALS[$i] ))

        # Quote N/A values, pass numbers bare
        _tps="${TPS_VALUES[$i]}"
        _tflops="${TFLOPS_VALUES[$i]}"
        _mfu="${MFU_VALUES[$i]}"
        if [[ "${_tps}" == "N/A" ]]; then _tps="null"; fi
        if [[ "${_tflops}" == "N/A" ]]; then _tflops="null"; fi
        if [[ "${_mfu}" == "N/A" ]]; then _mfu="null"; fi

        _mem="${MEMORY_VALUES[$i]}"
        if [[ "${_mem}" == "N/A" ]]; then _mem="null"; else _mem="\"${_mem}\""; fi

        _wandb="${WANDB_VALUES[$i]}"
        if [[ -z "${_wandb}" ]]; then _wandb="null"; else _wandb="\"${_wandb}\""; fi

        comma=","
        if ((i == NUM_CONFIGS - 1)); then comma=""; fi

        echo "    {"
        echo "      \"model\": \"${LABELS[$i]}\","
        echo "      \"module\": \"${MODULES[$i]}\","
        echo "      \"config\": \"${CONFIGS[$i]}\","
        echo "      \"tp\": ${tp},"
        echo "      \"dp\": ${dp},"
        echo "      \"lbs\": ${LBS_VALS[$i]},"
        echo "      \"gas\": ${GAS_VALS[$i]},"
        echo "      \"gbs\": ${gbs},"
        echo "      \"seq_len\": ${SEQ_VALS[$i]},"
        echo "      \"tps\": ${_tps},"
        echo "      \"tflops\": ${_tflops},"
        echo "      \"mfu\": ${_mfu},"
        echo "      \"memory\": ${_mem},"
        echo "      \"status\": \"${STATUSES[$i]}\","
        echo "      \"wall_s\": ${WALL_TIMES[$i]},"
        echo "      \"wandb\": ${_wandb}"
        echo "    }${comma}"
    done
    echo "  ]"
    echo "}"
} >"${RESULTS_JSON}"

# ---------------------------------------------------------------------------
# Print report to stdout
# ---------------------------------------------------------------------------
echo "============================================================"
cat "${REPORT}"
echo "============================================================"
echo ""
echo "Report saved to: ${REPORT}"
echo "Results JSON:    ${RESULTS_JSON}"
echo "Logs saved to:   ${OUTDIR}/"
