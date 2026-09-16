#!/bin/bash --login
# In-allocation runner for the 80B head-to-head convergence comparison at the
# production batch (GBS=6144). Companion to run_lr_finder.sh: same validated
# 80B stable corner (TP=4 / LBS=1 / GAS-derived GBS=6144 / compile OFF /
# AC=full / pure FSDP), but instead of sweeping LR it trains each optimizer at
# its finder-recommended constant LR for CONV_STEPS steps, so we can compare
# loss trajectories past the finder's early-step window.
#
# Recommendations from the 2026-06-27 production-batch finder
# (docs/experiments/lr-finder/agpt/80b/README.md):
#   mano    @ ~3e-6  (min/5; clean U-min at 1.6e-5, safest)
#   sophiag @ ~1e-6  (min/2.5; real U-min at 2.5e-6, narrower band)
#   adamw   @ ~5e-7  (under the 7.4e-7 NaN cliff)
#
# Knobs (qsub -v):
#   CONV_OPTIMIZERS   space-separated   (default "mano sophiag adamw")
#   CONV_STEPS        training steps    (default 200; smoke uses ~10)
#   CONV_GBS          global batch      (default 6144)
#   CONV_DUMP_FOLDER  output dir        (default outputs/conv-80b-gbs6144)
#   CONV_DATA_CACHE_PATH  shared warm blendcorpus index (recommended)
#   CONV_IDLE_TIMEOUT     launch idle timeout secs (default 2400)
#   CONV_TIMEOUT          per-optimizer hard wall secs (default 10800)
#   CONV_LR_<opt>     per-optimizer LR override (e.g. CONV_LR_mano=3e-6)
#   CONV_WARMUP       warmup steps      (default 10; small so we measure the
#                     target LR, not the ramp -- constant after warmup)

set -o pipefail

# ---------------------------------------------------------------------------
# Environment setup -- mirrors run_lr_finder.sh exactly. The bare `ezpz
# launch` inside the per-optimizer loop only resolves if the broadcast
# /tmp/.venv is active on every node; skipping the yeet-env + /tmp/.venv
# activation is what caused the first smoke (job 12469907) to exit 127 with
# "env: 'ezpz': No such file or directory".
# ---------------------------------------------------------------------------
module load oneapi/release/2025.3.1 hdf5 pti-gpu
# /opt/pbs/bin on PATH so `sh.qstat` works inside `ezpz launch`
# (ezpz.pbs.get_pbs_jobid_of_active_job calls `from sh import qstat`).
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

# Must cd into the repo BEFORE ezpz_setup_job and before sourcing .venv
# (PBS spawns in $HOME; a relative `source .venv/...` would pick up
# $HOME/.venv, and ezpz_setup_job captures WORKING_DIR=$(pwd) at start).
cd "${PBS_O_WORKDIR:-$(pwd)}"

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
set -u

source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "conv-80b: yeet-env via tarball (.venv.tar.gz)"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "conv-80b: yeet-env via per-file rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

CONV_OPTIMIZERS="${CONV_OPTIMIZERS:-mano sophiag adamw}"
CONV_STEPS="${CONV_STEPS:-200}"
CONV_GBS="${CONV_GBS:-6144}"
CONV_DUMP_FOLDER="${CONV_DUMP_FOLDER:-outputs/conv-80b-gbs6144}"
CONV_IDLE_TIMEOUT="${CONV_IDLE_TIMEOUT:-2400}"
CONV_TIMEOUT="${CONV_TIMEOUT:-10800}"
CONV_WARMUP="${CONV_WARMUP:-10}"

# Finder-recommended constant LRs (overridable via CONV_LR_<opt>).
declare -A LR=( [mano]="3e-6" [sophiag]="1e-6" [adamw]="5e-7" )
for opt in "${!LR[@]}"; do
    var="CONV_LR_${opt}"
    [[ -n "${!var:-}" ]] && LR[$opt]="${!var}"
done

# NGPUS / NHOSTS / NGPU_PER_HOST are exported by ezpz_setup_job above.
NGPU_PER_HOST="${NGPU_PER_HOST:-12}"
NGPUS="${NGPUS:-$(( ${NHOSTS:-1} * NGPU_PER_HOST ))}"
DATASET_PATH="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt"

echo "============================================================"
echo " 80B convergence comparison @ GBS=${CONV_GBS}"
echo " optimizers : ${CONV_OPTIMIZERS}"
echo " steps      : ${CONV_STEPS} (warmup ${CONV_WARMUP}, constant after)"
echo " devices    : ${NGPUS}"
echo " git HEAD   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "============================================================"

pkill -u "${USER}" -f "torchtitan.experiments.ezpz.train" 2>/dev/null && sleep 2 || true

read -ra OPTS <<< "${CONV_OPTIMIZERS}"
for opt in "${OPTS[@]}"; do
    lr="${LR[$opt]:-1e-6}"
    label="80b/${opt}@${lr}"
    logdir="${CONV_DUMP_FOLDER}/${opt}"
    mkdir -p "${logdir}"
    logfile="${logdir}/train.log"
    echo ""
    echo "--- [${label}] convergence (steps=${CONV_STEPS}) ---"
    echo "    logfile: ${logfile}"

    cache_args=()
    [[ -n "${CONV_DATA_CACHE_PATH:-}" ]] && \
        cache_args=(--dataloader.data_cache_path "${CONV_DATA_CACHE_PATH}/${opt}")

    # 80B validated stable corner (see run_lr_finder.sh / 80b production docs).
    # Constant LR: warmup CONV_WARMUP steps then flat (decay_ratio 0 ->
    # no decay phase, min_lr_factor 1.0 -> hold peak). AC subcommand is a
    # positional tyro token and MUST be last.
    timeout "${CONV_TIMEOUT}" \
        stdbuf -oL -eL \
        env NGPU="${NGPUS}" PYTHONUNBUFFERED=1 \
        ezpz launch \
        --nproc "${NGPUS}" \
        --nproc_per_node "${NGPU_PER_HOST}" \
        --auto-retry \
        --spare-nodes auto \
        --timeout "${CONV_IDLE_TIMEOUT}" \
        -- \
        python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config agpt_80b \
        --job.dump-folder "${CONV_DUMP_FOLDER}" \
        --optimizer "${opt}" \
        --optimizer.lr "${lr}" \
        --training.steps "${CONV_STEPS}" \
        --training.local_batch_size 1 \
        --training.global_batch_size "${CONV_GBS}" \
        --training.seq_len 8192 \
        --lr_scheduler.warmup_steps "${CONV_WARMUP}" \
        --lr_scheduler.decay_ratio 0.0 \
        --lr_scheduler.min_lr_factor 1.0 \
        --metrics.log_freq 1 \
        --metrics.enable_wandb \
        --checkpoint.no-enable \
        --dataloader.dataset blendcorpus \
        --dataloader.dataset_path "${DATASET_PATH}" \
        "${cache_args[@]}" \
        --parallelism.tensor_parallel_degree 4 \
        --parallelism.expert_parallel_degree 1 \
        --parallelism.data_parallel_replicate_degree 1 \
        --parallelism.data_parallel_shard_degree -1 \
        --compile.no-enable \
        activation-checkpoint:full \
        >"${logfile}" 2>&1 || echo "    [${label}] exit=$? (see logfile)"
    echo "    [${label}] done"
done
echo ""
echo "all convergence runs complete -> ${CONV_DUMP_FOLDER}"
