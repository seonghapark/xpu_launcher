#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-2b-failover
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
# select= overridden via qsub -l select=N (e.g. 522 = 512 train + 10 spare)
#PBS -q prod
#PBS -j oe

# Bad-node failover variant of submit_agpt_20b_aurora_venv.sh.
#
# Submit with:
#   qsub -l select=522 -l walltime=12:00:00 \
#     -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=3 \
#     scripts/submit_agpt_20b_aurora_venv_failover.sh
#
# The job requests 522 nodes from PBS, splits the nodelist into 512
# active + 10 spare, runs training on the active subset, and on a
# bad-node crash swaps the offending node out for a spare and
# retries. Up to FAILOVER_MAX_RETRIES retries (default 3).
#
# Required: NHOSTS_TRAIN env var (number of nodes to actually train on).
# Optional: FAILOVER_MAX_RETRIES (default 3).
# All other vars (LBS, GAS, OPTIMIZER, LR, CKPT_DIR, etc.) match the
# baseline submit_agpt_20b_aurora_venv.sh.

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN env var required (use qsub -v NHOSTS_TRAIN=N)"
    exit 1
fi

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# Source ezpz-utils + failover lib
# NOTE: PBS copies the submit script into /var/spool/pbs/mom_priv/jobs/, so
# `$(dirname "$(realpath "$0")")` is NOT the original scripts dir. The only
# reliable way to find sibling files is via $PBS_O_WORKDIR (the dir from
# which qsub was run) + the canonical relative path under the repo.
SCRIPTS_DIR="${PBS_O_WORKDIR:-$PWD}/torchtitan/experiments/ezpz/scripts"
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

# Source failover lib (sets up failover_init / failover_yeet_all / failover_run).
FAILOVER_LIB="$SCRIPTS_DIR/failover_lib.sh"
[[ -f "$FAILOVER_LIB" ]] || { echo "ERROR: failover_lib.sh not found at $FAILOVER_LIB (PBS_O_WORKDIR=$PBS_O_WORKDIR, PWD=$PWD)"; exit 1; }
source "$FAILOVER_LIB"

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Split PBS_NODEFILE into active (NHOSTS_TRAIN nodes) + spare (rest).
# This MUST happen before ezpz_setup_job so NHOSTS auto-derives correctly.
failover_init "$NHOSTS_TRAIN" || exit 1

# Now ezpz_setup_job sees the active subset only.
ezpz_setup_job

source .venv/bin/activate
# yeet to ALL nodes (active + spare) so any spare can swap in instantly.
# SKIP_YEET=1 lets an outer orchestrator (the multi-chain umbrella) pre-stage
# the venv itself and skip this per-job broadcast -- needed when several jobs
# share ONE PBS allocation, since ezpz yeet's local-copy always lands on the
# shared mom node's /tmp and concurrent broadcasts would corrupt each other.
# VENV_DST is where the (pre-staged or just-yeeted) venv lives on each node;
# defaults to /tmp/.venv so standalone behavior is unchanged.
if [[ "${SKIP_YEET:-0}" != "1" ]]; then
    failover_yeet_all || exit 1
fi
deactivate
source "${VENV_DST:-/tmp/.venv}/bin/activate"

# Kill stale palsd processes from previous runs.
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

# ---- Configuration (matches baseline submit_agpt_20b_aurora_venv.sh) ----
MODEL="2b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-2}"
GAS="${GAS:-1}"
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS="${TRAINING_STEPS:-$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))}"

OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

# CKPT_KEEP_LATEST_K is HARDCODED to 0 (= keep all). DO NOT change.
# Setting this > 0 causes torchtitan's _purge_stale_checkpoints() to
# delete every prior step-* dir on every save, irreversibly. Lost ~334
# chain ckpts on 2026-05-25 from one accidental `-v CKPT_KEEP_LATEST_K=10`
# override (job 8505252). If you legitimately need rotation, use a
# fresh CKPT_DIR override on a separate experiment — not the canonical chain.
if [[ -n "${CKPT_KEEP_LATEST_K:-}" && "${CKPT_KEEP_LATEST_K}" != "0" ]]; then
    echo "ERROR: CKPT_KEEP_LATEST_K=${CKPT_KEEP_LATEST_K} is set but this script hardcodes it to 0." >&2
    echo "       Setting keep_latest_k>0 will destroy older ckpts in the canonical chain." >&2
    echo "       If you really want rotation, use a fresh CKPT_DIR override on a separate experiment." >&2
    exit 1
fi
CKPT_KEEP_LATEST_K=0
CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}}"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} (failover wrapper, ${NNODES} active nodes)"
log_message INFO "-------------------------------------------"
log_message INFO "PBS allocation: $(wc -l < "$FAILOVER_PBS_NODEFILE_ORIG") nodes total"
log_message INFO "  active: $(wc -l < "$FAILOVER_ACTIVE")  spare: $(wc -l < "$FAILOVER_SPARE")"
log_message INFO "FAILOVER_MAX_RETRIES: ${FAILOVER_MAX_RETRIES:-3}"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "GBS: ${GBS}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Preflight: catch bad nodes BEFORE 30+ min of model init ----
# Run a tiny single-rank-per-node ezpz.examples.test through failover_run so
# that any bad nodes are detected (gloo/UR/SIGSEGV/timeout) and swapped for
# spares before the real training command launches.
#
# Timeout sizing: ~40s actual training on 8N; DDP init/all-reduce dominates
# at scale. 8505298 (8N) needed ~120s; 8506215 (512N) tripped 120s watchdog
# during DDP init at 6144 ranks. 600s (10 min) leaves plenty of headroom
# for 1024N+ while still catching genuine hangs quickly.
# Requires ezpz >= 0.16.0 for --timeout.
# --train-iters 5: enough to verify all-reduce works without burning hours of
# walltime at large N. Each iter at 6144 ranks (512N) is ~4-5 min, so 5 iters
# ≈ 25-30 min including DDP init. Without this, the test defaults to 200 iters
# (~16h at 512N — full walltime burned in preflight).
log_message INFO "preflight smoke: ezpz.examples.test on active nodes"
FAILOVER_IDLE_TIMEOUT="${PREFLIGHT_IDLE_TIMEOUT:-600}" FAILOVER_MAX_RETRIES=2 \
    failover_run ezpz launch python3 -m ezpz.examples.test --train-iters 5 \
    || { log_message ERROR "preflight smoke failed after retries; bailing"; exit 1; }
log_message INFO "preflight smoke OK — proceeding to main training launch"

# ---- Walltime-aware checkpointing ----
# Force a checkpoint before the PBS walltime runs out (a short job can otherwise
# save nothing). Export an ABSOLUTE deadline (job_start + walltime) as
# WALLTIME_DEADLINE_EPOCH -- it survives failover relaunches, so the backstop
# still fires after a mid-job bad-node swap (the relative WALLTIME_SECONDS reset
# on retry and the job was SIGTERM'd mid-save, 2026-06-27). Also export the
# relative remaining as a fallback. config_registry reads both; mpiexec --envall
# propagates them to all ranks. Override either in the environment (0 to disable).
if [[ -z "${WALLTIME_DEADLINE_EPOCH:-}" ]]; then
    WALLTIME_DEADLINE_EPOCH=$(failover_deadline_epoch)
fi
if [[ -z "${WALLTIME_SECONDS:-}" ]]; then
    WALLTIME_SECONDS=$(failover_remaining_walltime)
fi
export WALLTIME_DEADLINE_EPOCH WALLTIME_SECONDS
log_message INFO "WALLTIME_DEADLINE_EPOCH: ${WALLTIME_DEADLINE_EPOCH} (remaining ~${WALLTIME_SECONDS}s)"

# ---- Launch with failover ----
failover_run ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}${CONFIG_SUFFIX:-}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}" \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
