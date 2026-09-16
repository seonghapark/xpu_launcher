#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N yeet-env-scaling-test
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#
# yeet-env tarball-broadcast scaling test.
#
# Records:
#   * Pre-yeet wall-clock timestamp
#   * Post-yeet wall-clock timestamp (script-side)
#   * The "Done in Xs" line that yeet-env itself prints
#   * Then runs 10 training steps to verify the broadcast venv works
#     end-to-end and emits per-iter throughput numbers.
#
# Submit per-N (debug-scaling for ≤256, small route for 512/1024,
# prod-large route for ≥1920):
#   qsub -l select=8   -l walltime=00:30:00 -q debug-scaling \
#       torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh
#
# Output is appended to .yeet-env-scaling-results.csv in the workdir
# (creates it with a header if missing). Use the file as the data
# source for any downstream plotting.

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

EZPZ_UTILS="$(dirname "$(realpath "$0")")/../../../.ezpz-utils-cache/ezpz-utils.sh"
EZPZ_UTILS="$(realpath "$EZPZ_UTILS" 2>/dev/null || echo "")"
if [[ -z "$EZPZ_UTILS" || ! -f "$EZPZ_UTILS" ]]; then
    EZPZ_UTILS="${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh"
fi
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"

NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
RESULTS_CSV="${RESULTS_CSV:-${PBS_O_WORKDIR}/.yeet-env-scaling-results.csv}"
if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "jobid,nodes,yeet_seconds,first_step_seconds,steady_tps_per_gpu,steady_tflops" > "${RESULTS_CSV}"
fi

# ---- yeet-env (timed) ----
source .venv/bin/activate
YEET_T0=$(date +%s.%N)
log_message INFO "yeet-env scaling: NNODES=${NNODES} starting yeet at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ -f .venv.tar.gz ]]; then
    ezpz yeet-env --src .venv.tar.gz | tee /tmp/yeet-env.${PBS_JOBID%%.*}.log
else
    log_message WARN "yeet-env scaling: no .venv.tar.gz, falling back to rsync (will be much slower)"
    ezpz yeet-env | tee /tmp/yeet-env.${PBS_JOBID%%.*}.log
fi
YEET_T1=$(date +%s.%N)
YEET_SECONDS=$(awk -v a="${YEET_T0}" -v b="${YEET_T1}" 'BEGIN{printf "%.1f", b-a}')
# Also extract the "Done in Xs" yeet-env reports internally.
YEET_INTERNAL=$(grep -oE "Done in [0-9.]+s" /tmp/yeet-env.${PBS_JOBID%%.*}.log | head -1 | grep -oE "[0-9.]+")
log_message INFO "yeet-env scaling: NNODES=${NNODES} yeet wall-clock=${YEET_SECONDS}s, yeet internal=${YEET_INTERNAL}s"
deactivate
source /tmp/.venv/bin/activate

# Stale palsd cleanup
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

# ---- Minimal 10-step training (2B, plain CE) ----
SEQ_LEN=8192
TP=1; PP=1; CP=1
LBS=2; GAS=1
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL="${DFL_PARENT}/olmo-mix-1124.txt"

# Per-job checkpoint dir to avoid step-0 ckpt collisions across the sweep.
CKPT_DIR="checkpoints/yeet-scaling/n${NNODES}-job${PBS_JOBID%%.*}"

TRAIN_T0=$(date +%s.%N)
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_2b \
    --no-checkpoint.enable \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${CKPT_DIR}/.cache" \
    --debug.print-config \
    --optimizer=sophiag \
    --optimizer.lr=2.28e-5 \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps=10 \
    "$@" 2>&1 | tee /tmp/train.${PBS_JOBID%%.*}.log
TRAIN_T1=$(date +%s.%N)

# First-step time (model build + first step)
FIRST_STEP_T=$(awk -v a="${TRAIN_T0}" -v b="${TRAIN_T1}" 'BEGIN{printf "%.1f", b-a}')

# Steady-state TPS/GPU & TFLOPS — use the median of steps 5-10.
STEADY_TPS=$(grep -oE "tps:\s*[0-9,]+" /tmp/train.${PBS_JOBID%%.*}.log | tr -d ',' | awk '{print $2}' | tail -6 | sort -n | awk 'NR==3{print}')
STEADY_TFLOPS=$(grep -oE "tflops:\s*[0-9.]+" /tmp/train.${PBS_JOBID%%.*}.log | awk '{print $2}' | tail -6 | sort -n | awk 'NR==3{print}')

echo "${PBS_JOBID%%.*},${NNODES},${YEET_INTERNAL:-${YEET_SECONDS}},${FIRST_STEP_T},${STEADY_TPS:-NA},${STEADY_TFLOPS:-NA}" \
    >> "${RESULTS_CSV}"

log_message INFO "yeet-env scaling: NNODES=${NNODES} done. Recorded to ${RESULTS_CSV}"
log_message INFO "Summary: yeet=${YEET_INTERNAL}s, first_step=${FIRST_STEP_T}s, tps=${STEADY_TPS}, tflops=${STEADY_TFLOPS}"
