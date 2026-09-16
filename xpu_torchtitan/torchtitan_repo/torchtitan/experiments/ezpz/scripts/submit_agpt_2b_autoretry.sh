#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-2b-autoretry
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -j oe
# select= overridden via qsub -l select=N (e.g. 14 = 12 train + 2 spare)

# Native-auto-retry variant of submit_agpt_2b_aurora_venv_failover.sh.
#
# This script uses `ezpz launch --auto-retry` EXCLUSIVELY for bad-node
# failover -- there is no scripts/failover_lib.sh involvement. Since
# ezpz PR #170 the launcher does natively what the old bash wrapper did:
# split the PBS allocation into active + spare, run the inner command,
# scrape the same bad-node signatures on any non-zero exit (incl. the
# idle-output watchdog exit 124 and walltime-racing crashes that surface
# as 143), swap a spare in-place, and retry until success / walltime /
# spare-exhaustion / stuck-pre-training / SIGINT.
#
# Portable: PBS headers default to Aurora (AuroraGPT / prod /
# home:flare). The data list also auto-selects by machine
# (olmo-mix-1124 on Aurora, books on Sunspot), so no body edit is needed
# to move between them.
#
# Submit (Aurora):
#   qsub -l select=522 -l walltime=12:00:00 -v NHOSTS_TRAIN=512 \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_2b_autoretry.sh
#
# Sunspot: override the #PBS directives at submit time -- qsub flags beat
# the #PBS lines:
#   qsub -A datascience -q workq -l filesystems=flare:home \
#     -l select=14 -l walltime=12:00:00 -v NHOSTS_TRAIN=12 \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_2b_autoretry.sh
#
# The job requests NHOSTS_TRAIN + spares nodes from PBS. ezpz splits the
# nodelist into NHOSTS_TRAIN active + (rest) spare via --spare-nodes auto;
# training runs on the active subset, and a bad-node crash swaps the
# offending node out for a spare and retries.
#
# Required: NHOSTS_TRAIN env var (number of nodes to actually train on).
# Optional: MAX_FAILOVER_RETRIES (default: unbounded -- governed by spare
#           count + walltime), IDLE_TIMEOUT (default 1800s), plus all the
#           training vars (LBS, GAS, OPTIMIZER, LR, CKPT_DIR, DFL_NAME,
#           CHECKPOINT_ASYNC_MODE, ...) which match the failover script.

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN env var required (use qsub -v NHOSTS_TRAIN=N)"
    exit 1
fi

# ---- Common Intel-XPU runtime env (machine-agnostic bits) ----
# Module loads are handled by ezpz_load_modules below (machine-aware), so
# we only set the runtime knobs here -- not the Aurora-specific
# `module load oneapi...` the failover script hardcoded.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# ---- Source ezpz-utils ----
# NOTE: PBS copies the submit script into /var/spool/pbs/mom_priv/jobs/, so
# `$(dirname "$(realpath "$0")")` is NOT the original scripts dir. Find
# sibling files via $PBS_O_WORKDIR (the dir from which qsub was run).
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

cd "${PBS_O_WORKDIR:-$(pwd)}"

# ---- Environment setup + venv broadcast ----
# ezpz_load_modules / ezpz_setup_job are machine-aware (dispatch on
# ezpz_get_machine_name), so this block is portable across Sunspot/Aurora.
#
# CRITICAL: `ezpz launch --auto-retry` does NOT broadcast the venv to spare
# nodes itself -- it only splits the nodelist + runs the retry loop. So we
# still yeet .venv.tar.gz to the FULL allocation here, exactly like the old
# failover_yeet_all did. Because we leave PBS_NODEFILE whole (we do NOT
# pre-split it -- ezpz does that internally from --nproc), a plain
# `ezpz yeet` covers active + spare automatically, so any swapped-in spare
# already has /tmp/.venv ready.
ezpz_load_modules
ezpz_setup_job
source .venv/bin/activate
# .venv.tar.gz -> extracted to /tmp/.venv on every node in PBS_NODEFILE.
if [[ -f .venv.tar.gz ]]; then
    ezpz yeet --src .venv.tar.gz
else
    ezpz yeet
fi
deactivate
# The driver must run from the broadcast venv so mpiexec points ranks at
# /tmp/.venv (node-local), not the Lustre .venv.
source /tmp/.venv/bin/activate

# Kill stale palsd processes from previous runs.
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

# ---- Active-rank arithmetic ----
# ezpz_setup_job sees the FULL allocation, so its $NGPUS counts active +
# spare. For GBS we need the ACTIVE rank count only. ezpz's --auto-retry
# keeps the active count constant across retries (a swap replaces a node
# in-place by index, never changing the count), so this is valid for the
# whole run; at runtime os.environ["WORLD_SIZE"] also equals NGPUS_ACTIVE.
PPN="${NGPU_PER_HOST:-12}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))   # == --nproc passed to ezpz launch
NNODES="$NHOSTS_TRAIN"                    # for CKPT_DIR naming

# ---- Configuration (matches submit_agpt_2b_aurora_venv_failover.sh) ----
MODEL="2b"
# Default to the `_real` flavor: real-valued (cos_sin) RoPE instead of the
# complex backend, which torch.compile's inductor cannot lower (it falls
# back to eager). With compile ON (the 2B default) cos_sin is faster.
# Override with CONFIG_SUFFIX= (empty) to get the plain complex flavor.
CONFIG_SUFFIX="${CONFIG_SUFFIX-_real}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-2}"
GAS="${GAS:-1}"
GBS=$(( NGPUS_ACTIVE * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS="${TRAINING_STEPS:-$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))}"

OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

# Validation: run the EzpzValidator on the blendcorpus validation split
# every VALIDATOR_FREQ steps for VALIDATOR_STEPS iters. On by default;
# set VALIDATOR_FREQ to a huge number or pass --validator.no-enable to skip.
VALIDATOR_FREQ="${VALIDATOR_FREQ:-100}"
VALIDATOR_STEPS="${VALIDATOR_STEPS:-10}"

# Per-machine data default. Aurora's canonical 2B mixture is olmo-mix-1124.
# On Sunspot that list does not exist AND the dolma list points at /gila
# (not mounted on Sunspot); the tokenized data actually resident on Sunspot
# (/flare .../books-dataset) is the `books` list, so that is the default here.
# DFL_NAME stays env-overridable.
MACHINE="$(ezpz_get_machine_name)"
case "$MACHINE" in
    aurora)  DEFAULT_DFL_NAME="olmo-mix-1124" ;;
    sunspot) DEFAULT_DFL_NAME="books" ;;
    *)       DEFAULT_DFL_NAME="books" ;;
esac
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/${MACHINE}"
DFL_NAME="${DFL_NAME:-$DEFAULT_DFL_NAME}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

# CKPT_KEEP_LATEST_K is HARDCODED to 0 (= keep all). DO NOT change.
# Setting this > 0 causes torchtitan's _purge_stale_checkpoints() to
# delete every prior step-* dir on every save, irreversibly. Lost ~334
# chain ckpts on 2026-05-25 from one accidental `-v CKPT_KEEP_LATEST_K=10`
# override (job 8505252). If you legitimately need rotation, use a
# fresh CKPT_DIR override on a separate experiment -- not the canonical chain.
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

# Validator flag set, gated by VALIDATOR_ENABLE (default 1 = on). Set
# VALIDATOR_ENABLE=0 to skip validation entirely (no --validator.enable,
# no validator dataloader build) -- e.g. for batch-size/throughput studies
# or to sidestep a validator collective issue at scale.
if [[ "${VALIDATOR_ENABLE:-1}" == "1" ]]; then
    VALIDATOR_FLAGS=(
        "--validator.enable"
        "--validator.freq=${VALIDATOR_FREQ}"
        "--validator.steps=${VALIDATOR_STEPS}"
        "--validator.dataloader.dataset-path=${DFL}"
        "--validator.dataloader.data-cache-path=${DATA_CACHE_PATH}"
    )
else
    VALIDATOR_FLAGS=("--validator.no-enable")
fi

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} (native --auto-retry, ${NNODES} active nodes)"
log_message INFO "-------------------------------------------"
log_message INFO "machine: ${MACHINE}"
log_message INFO "PBS allocation: $(wc -l < "${PBS_NODEFILE}") nodes total"
log_message INFO "  active (NHOSTS_TRAIN): ${NHOSTS_TRAIN}  spare: --spare-nodes auto (rest)"
log_message INFO "--nproc (active ranks): ${NGPUS_ACTIVE}  -ppn: ${PPN}"
log_message INFO "MAX_FAILOVER_RETRIES: ${MAX_FAILOVER_RETRIES:-unbounded}"
log_message INFO "IDLE_TIMEOUT: ${IDLE_TIMEOUT:-1800}"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "GBS: ${GBS}"
log_message INFO "data list: ${DFL}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Launch with native auto-retry ----
# No preflight: ezpz's STUCK_PRE_TRAINING guard already bails (without
# burning spares) if init crashes twice with zero training progress.
# --spare-nodes auto => spares = total_pbs_nodes - NHOSTS_TRAIN, so the
# submitter sets `select = NHOSTS_TRAIN + desired_spares`.
#
# Build the optional --max-failover-retries as an ARRAY, not an inline
# ${VAR:+--flag "$VAR"} -- the latter collapses to a single argv token
# ("--max-failover-retries 3") that argparse rejects. The array expands
# to two tokens when set and zero tokens when unset.
mfr_args=()
[[ -n "${MAX_FAILOVER_RETRIES:-}" ]] && mfr_args=(--max-failover-retries "${MAX_FAILOVER_RETRIES}")

ezpz launch \
    --nproc "${NGPUS_ACTIVE}" \
    --nproc_per_node "${PPN}" \
    --auto-retry \
    --spare-nodes auto \
    --timeout "${IDLE_TIMEOUT:-1800}" \
    "${mfr_args[@]}" \
    -- \
    python3 -m torchtitan.experiments.ezpz.train \
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
    "${VALIDATOR_FLAGS[@]}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
