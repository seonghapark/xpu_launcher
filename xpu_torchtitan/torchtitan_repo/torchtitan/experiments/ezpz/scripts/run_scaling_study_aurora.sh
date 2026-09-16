#!/bin/bash --login
# Aurora scaling study benchmark for ezpz experiment configs.
# Runs a configurable group of models for BENCH_STEPS iterations each,
# parses metrics, and generates report.md + results.json for aggregation.
#
# Designed for Aurora's queue structure:
#   - debug-scaling (1h): split into light/heavy groups
#   - prod (12h): run all configs combined
#
# Usage (inside a PBS allocation):
#   SCALING_GROUP=light bash torchtitan/experiments/ezpz/scripts/run_scaling_study_aurora.sh
#   SCALING_GROUP=heavy bash torchtitan/experiments/ezpz/scripts/run_scaling_study_aurora.sh
#   SCALING_GROUP=all   bash torchtitan/experiments/ezpz/scripts/run_scaling_study_aurora.sh
#
# Environment variables:
#   SCALING_GROUP   — which configs to run: light|heavy|all (default: all)
#   BENCH_STEPS     — training iterations per run (default: 20)
#   BENCH_TIMEOUT   — per-run timeout in seconds (default: 2400)
#   FILTER_NONZERO_RANKS — set to 1 to suppress non-rank-0 output (default: 0)
#   SCALING_OUTDIR  — override output directory

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup — mirror production submit_agpt_*_aurora_venv.sh:
#   1. CCL/oneAPI env vars (matches submit_agpt_2b_aurora_venv_failover.sh)
#   2. ezpz-utils + ezpz_setup_job
#   3. activate project .venv (torch 2.13)
#   4. yeet .venv to local /tmp on every compute node (REQUIRED at N >= 256;
#      without it, all ranks load Python from flare and saturate metadata,
#      triggering set_determinism std::bad_alloc/SIGSEGV crashes at 6,144+
#      ranks — exactly what hits production 512N intermittently)
#   5. reactivate /tmp/.venv so subsequent ezpz launch uses local python
# ---------------------------------------------------------------------------
set +u
# When this script is invoked via `qsub -- /bin/bash -c 'bash <this>'`,
# the inner bash is NOT a login shell, so module/MODULEPATH/lmod aren't
# initialized. Source Aurora's Cray PE init (which defines `module` AND
# populates MODULEPATH from /etc/cray-pe.d/cray-pe-configuration.sh).
# `/etc/bash.bashrc.local` is what `#!/bin/bash --login` gets via
# /etc/bash.bashrc -> /etc/bash.bashrc.local.
if ! command -v module >/dev/null 2>&1 || [[ -z "${MODULEPATH:-}" ]]; then
    if [[ -r /etc/bash.bashrc.local ]]; then
        source /etc/bash.bashrc.local
    elif [[ -r /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi
module load oneapi/release/2025.3.1 hdf5 pti-gpu
# /opt/pbs/bin must be on PATH so `sh.qstat` works inside `ezpz launch`
# (ezpz.pbs.get_pbs_jobid_of_active_job calls `from sh import qstat`).
# bash --login on login node has it via /etc/profile; this re-export
# ensures it propagates through mpiexec --envall to compute nodes too.
export PATH="/opt/pbs/bin:${PATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# Must cd into the repo BEFORE ezpz_setup_job (and before sourcing .venv).
# Reasons (both bite us in different ways):
#   1. PBS spawns scripts in $HOME, so `source .venv/bin/activate` (relative)
#      would pick up $HOME/.venv if present, with incompatible torch.
#   2. ezpz_setup_job's `WORKING_DIR=$(pwd)` runs at script start. If we
#      call ezpz_setup_job before this cd, WORKING_DIR captures $HOME, then
#      its "WORKING_DIR doesn't match PBS_O_WORKDIR" branch OVERWRITES
#      PBS_O_WORKDIR with $HOME (yes, the opposite of what you'd want).
#      Subsequent `cd "$PBS_O_WORKDIR"` then lands in $HOME, defeating the
#      cd entirely. cd FIRST → WORKING_DIR captures the right path → no
#      overwrite needed.
cd "${PBS_O_WORKDIR:-$(pwd)}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "scaling: yeet-env via tarball (.venv.tar.gz)"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "scaling: yeet-env via per-file rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate
set -u

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCALING_GROUP="${SCALING_GROUP:-all}"
BENCH_STEPS="${BENCH_STEPS:-20}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-2400}"
FILTER_NONZERO_RANKS="${FILTER_NONZERO_RANKS:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_NODES="${NHOSTS:-${SLURM_NNODES:-1}}"
DEVICES_PER_NODE=$(( NGPUS / NUM_NODES ))

# Append group suffix to output dir when not "all" to avoid collisions
# between light and heavy jobs at the same node count
_outdir_base="${SCALING_OUTDIR:-outputs/scaling_study_aurora/${TIMESTAMP}/n${NUM_NODES}}"
if [[ "${SCALING_GROUP}" != "all" ]]; then
    OUTDIR="${_outdir_base}/${SCALING_GROUP}"
else
    OUTDIR="${_outdir_base}"
fi
mkdir -p "${OUTDIR}"

DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

# ---------------------------------------------------------------------------
# Model configurations by group
#
# Group "light": fast, reliable at all node counts
#   agpt_2b, agpt_20b, moe_2b
#
# Group "heavy": slower, some failures at extremes
#   agpt_80b_wide (TP=4, skip <4N), moe_7b (OOMs at 32N+), moe_10b_2b
# ---------------------------------------------------------------------------
declare -a LABELS MODULES CONFIGS LBS_VALS TP_VALS GAS_VALS SEQ_VALS \
           COMPILE_VALS AC_VALS

_add_config() {
    LABELS+=("$1")
    MODULES+=("$2")
    CONFIGS+=("$3")
    LBS_VALS+=("$4")
    TP_VALS+=("$5")
    GAS_VALS+=("$6")
    SEQ_VALS+=("$7")
    COMPILE_VALS+=("$8")
    AC_VALS+=("$9")
}

if [[ "${SCALING_GROUP}" == "light" || "${SCALING_GROUP}" == "all" ]]; then
    #              label       module      config     lbs tp gas  seq   compile ac
    # LBS chosen to match production submit scripts (submit_agpt_{2b,20b}_aurora_venv.sh):
    # both agpt_2b and agpt_20b production runs use LBS=2 (GBS = N*12*2).
    _add_config    "agpt_2b"   "ezpz.agpt" "agpt_2b"   2   1   1  8192  "on"    "full"
    _add_config    "agpt_20b"  "ezpz.agpt" "agpt_20b"  2   1   1  8192  "on"    "full"
    _add_config    "moe_2b"    "ezpz.moe"  "moe_2b"    2   1   1  4096  "off"   "full"
fi

if [[ "${SCALING_GROUP}" == "heavy" || "${SCALING_GROUP}" == "all" ]]; then
    # Skip 80B_wide at <4 nodes (TP=4 needs at least 48 GPUs)
    if ((NUM_NODES >= 4)); then
        _add_config "agpt_80b_wide" "ezpz.agpt" "agpt_80b_wide" 1 4 1 8192 "on" "full"
    else
        echo "--- [agpt_80b_wide] SKIPPED: needs >=4 nodes for TP=4 (have ${NUM_NODES}) ---"
        echo ""
    fi
    _add_config    "moe_7b"      "ezpz.moe"  "moe_7b"         2   1   1  4096  "off"   "none"
    _add_config    "moe_10b_2b"  "ezpz.moe"  "moe_10b_2b_sdpa" 1  1   1  4096  "off"   "none"
fi

# FILTER_LABELS: space-separated label whitelist (e.g. "agpt_2b"). When set,
# drops all other configs before benchmarking. Used by single-model scaling
# sweeps like submit_2b_scaling_aurora.sh.
if [[ -n "${FILTER_LABELS:-}" ]]; then
    read -ra _keep <<< "${FILTER_LABELS}"
    declare -a _F_LABELS _F_MODULES _F_CONFIGS _F_LBS _F_TP _F_GAS _F_SEQ _F_COMPILE _F_AC
    for ((i = 0; i < ${#LABELS[@]}; i++)); do
        for k in "${_keep[@]}"; do
            if [[ "${LABELS[$i]}" == "$k" ]]; then
                _F_LABELS+=("${LABELS[$i]}")
                _F_MODULES+=("${MODULES[$i]}")
                _F_CONFIGS+=("${CONFIGS[$i]}")
                _F_LBS+=("${LBS_VALS[$i]}")
                _F_TP+=("${TP_VALS[$i]}")
                _F_GAS+=("${GAS_VALS[$i]}")
                _F_SEQ+=("${SEQ_VALS[$i]}")
                _F_COMPILE+=("${COMPILE_VALS[$i]}")
                _F_AC+=("${AC_VALS[$i]}")
                break
            fi
        done
    done
    LABELS=("${_F_LABELS[@]}")
    MODULES=("${_F_MODULES[@]}")
    CONFIGS=("${_F_CONFIGS[@]}")
    LBS_VALS=("${_F_LBS[@]}")
    TP_VALS=("${_F_TP[@]}")
    GAS_VALS=("${_F_GAS[@]}")
    SEQ_VALS=("${_F_SEQ[@]}")
    COMPILE_VALS=("${_F_COMPILE[@]}")
    AC_VALS=("${_F_AC[@]}")
fi

# LBS_OVERRIDE_<label>: override LBS per label (e.g. LBS_OVERRIDE_agpt_2b=2 to
# match production GBS scaling). Lets one row be tuned without rewriting the
# add_config block.
for ((i = 0; i < ${#LABELS[@]}; i++)); do
    _label="${LABELS[$i]}"
    _var="LBS_OVERRIDE_${_label}"
    _override="${!_var:-}"
    if [[ -n "${_override}" ]]; then
        echo "--- LBS override: ${_label}  ${LBS_VALS[$i]} -> ${_override} ---"
        LBS_VALS[$i]="${_override}"
    fi
done

NUM_CONFIGS="${#LABELS[@]}"

if ((NUM_CONFIGS == 0)); then
    echo "ERROR: No configs selected (SCALING_GROUP=${SCALING_GROUP})"
    exit 1
fi

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
echo " Aurora Scaling Study — ${TIMESTAMP}"
echo " group=${SCALING_GROUP}  steps=${BENCH_STEPS}  devices=${NGPUS}  nodes=${NUM_NODES}"
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
    do_compile="${COMPILE_VALS[$i]}"
    ac_mode="${AC_VALS[$i]}"
    logfile="${OUTDIR}/${label}.log"

    # Compute GBS: DP = NGPUS / TP, GBS = DP * LBS * GAS
    dp=$(( NGPUS / tp ))
    gbs=$(( dp * lbs * gas ))

    echo "--- [${label}] running (module=${module} config=${config}) ---"
    echo "    TP=${tp} DP=${dp} LBS=${lbs} GAS=${gas} GBS=${gbs} seq_len=${seq_len}"
    echo "    compile=${do_compile} ac=${ac_mode}"
    echo "    started @ $(date +%H:%M:%S)"
    echo "    logfile: ${logfile}"
    echo ""

    start_seconds=$SECONDS

    # Build extra args
    extra_args=()
    if [[ "${do_compile}" == "off" ]]; then
        extra_args+=("--compile.no-enable")
    fi
    # 57th sync (PR #3674): AC is now a tyro subcommand union, not a
    # `--activation_checkpoint.mode=<str>` flag. The subcommand token
    # (`activation-checkpoint:<policy>`) is positional and must come
    # AFTER all --flags, so it goes last in the launch arg list (see
    # ac_subcommand below, appended after extra_args).
    case "${ac_mode}" in
        none)      ac_subcommand="activation-checkpoint:none" ;;
        full)      ac_subcommand="activation-checkpoint:full" ;;
        selective) ac_subcommand="activation-checkpoint:selective" ;;
        *) echo "Unknown ac_mode: ${ac_mode}" >&2; exit 1 ;;
    esac

    # Dataset selection: SCALING_DATASET=blendcorpus (default) uses local
    # books.txt list; SCALING_DATASET=<hf/repo> streams from HF.
    dataset_args=()
    if [[ "${SCALING_DATASET:-blendcorpus}" == "blendcorpus" ]]; then
        dataset_args+=("--dataloader.dataset" "blendcorpus"
                       "--dataloader.dataset_path" "${DATASET_PATH}")
    else
        dataset_args+=("--dataloader.dataset" "${SCALING_DATASET}")
    fi

    # Use absolute python3 from the activated /tmp/.venv to avoid PATH-order
    # races where mpiexec --envall propagates a stale PATH and rank-N python
    # resolves to the system python (which doesn't have ezpz).
    PY="${VIRTUAL_ENV:-/tmp/.venv}/bin/python3"

    timeout "${BENCH_TIMEOUT}" \
        stdbuf -oL -eL \
        env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
        ezpz launch "${PY}" -m torchtitan.experiments.ezpz.train \
        --module "${module}" \
        --config "${config}" \
        --training.steps "${BENCH_STEPS}" \
        --training.local_batch_size "${lbs}" \
        --training.global_batch_size "${gbs}" \
        --training.seq_len "${seq_len}" \
        --parallelism.tensor_parallel_degree "${tp}" \
        --metrics.log_freq 1 \
        --checkpoint.no-enable \
        "${dataset_args[@]}" \
        "${extra_args[@]}" \
        "${ac_subcommand}" \
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
    echo "# Aurora Scaling Study — ${NUM_NODES} Nodes (${SCALING_GROUP})"
    echo ""

    # Metadata table
    _meta_keys=("Date" "Commit" "Machine" "Job ID" "Nodes" "Devices" "Devices/Node" "Steps" "Group")
    _meta_vals=("${RUN_DATE}" "${GIT_COMMIT}" "${MACHINE_NAME}" "${JOB_ID}" "${NUM_NODES}" "${NGPUS}" "${DEVICES_PER_NODE}" "${BENCH_STEPS}" "${SCALING_GROUP}")
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
    printf "| %-${cw}s | %3s | %3s | %5s | %5s | %4s | %7s | %20s | %9s | %8s | %6s | %10s |" \
        "Config" "TP" "LBS" "GAS" "GBS" "AC" "Compile" "Memory" "TPS" "TFLOPS" "MFU" "Wall (s)"
    if $has_wandb; then printf " %-6s |" "W&B"; fi
    printf " %-6s |\n" "Status"

    # Separator
    _sep() { printf '%0.s-' $(seq 1 "$1"); }
    printf "|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|-%s-|" \
        "$(_sep "${cw}")" "$(_sep 3)" "$(_sep 3)" "$(_sep 5)" "$(_sep 5)" \
        "$(_sep 4)" "$(_sep 7)" \
        "$(_sep 20)" "$(_sep 9)" "$(_sep 8)" "$(_sep 6)" "$(_sep 10)"
    if $has_wandb; then printf -- "-%s-|" "$(_sep 6)"; fi
    printf -- "-%s-|\n" "$(_sep 6)"

    # Rows
    for ((i = 0; i < NUM_CONFIGS; i++)); do
        tp="${TP_VALS[$i]}"
        dp=$(( NGPUS / tp ))
        gbs=$(( dp * LBS_VALS[$i] * GAS_VALS[$i] ))
        printf "| %-${cw}s | %3s | %3s | %5s | %5s | %4s | %7s | %20s | %9s | %8s | %6s | %10s |" \
            "${LABELS[$i]}" \
            "${tp}" \
            "${LBS_VALS[$i]}" \
            "${GAS_VALS[$i]}" \
            "${gbs}" \
            "${AC_VALS[$i]}" \
            "${COMPILE_VALS[$i]}" \
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
    echo "  \"group\": \"${SCALING_GROUP}\","
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
        echo "      \"compile\": \"${COMPILE_VALS[$i]}\","
        echo "      \"ac\": \"${AC_VALS[$i]}\","
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
