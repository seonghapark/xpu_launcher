#!/bin/bash --login
#PBS -A datascience
#PBS -N moe-ab-check
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -j oe
#
# Generic A/B check for an MoE config across two git commits.
# Runs the same smoke twice (BASELINE vs HEAD), extracts loss +
# grad_norm per step, diffs them, and reports BIT-IDENTICAL / DRIFT.
#
# Uses ephemeral `git worktree add` checkouts so neither phase
# touches the main working tree. The shared `.venv` + `assets/hf`
# are symlinked into each worktree (no re-yeet).
#
# Usage from a login node:
#   qsub -l select=N -v BASELINE_COMMIT=<sha>,HEAD_COMMIT=<sha>,CONFIG=<name>,STEPS=<n>,EP=<n>,PADDING=<0|1> \
#       torchtitan/experiments/ezpz/scripts/moe_ab_check.sh
#
# Required env vars:
#   BASELINE_COMMIT  git ref for the baseline phase (branch, tag, SHA)
#   HEAD_COMMIT      git ref for the head phase
#   CONFIG           MoE config registry name (e.g. moe_debugmodel_ep,
#                    moe_10b_2b_sdpa_ep)
#
# Optional env vars:
#   STEPS            default 10
#   EP               override --parallelism.expert-parallel-degree
#   PADDING          0 or 1 (default 0) — sets TT_MOE_NORMAL_EQUAL_A2A_PADDING
#
# Output: logs/moe-ab-${CONFIG}-ep${EP}-pad${PADDING}-N${NNODES}-${JOBID}/
#   baseline.log, head.log, baseline.metrics, head.metrics, diff.txt,
#   verdict (BIT-IDENTICAL | DRIFT), run.log

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Bump XCCL IPC handle cache from default 1000 to 8000. At EP=12 +
# padding=1 + MoE A2A the default can fill up within 1-2 steps; evicted
# handles point into freed GPU memory and the next dispatch segfaults
# with `Segmentation fault from GPU at 0x... ctx_id: 1 (CCS) type: 0
# (NotPresent)`. Increasing the cache lets dispatch survive past
# step 1 so the A/B comparison can actually run.
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD="${CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD:-8000}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

CONFIG="${CONFIG:?CONFIG env var must be set (e.g. moe_debugmodel_ep)}"
BASELINE_COMMIT="${BASELINE_COMMIT:?BASELINE_COMMIT env var must be set (git ref)}"
HEAD_COMMIT="${HEAD_COMMIT:?HEAD_COMMIT env var must be set (git ref)}"
STEPS="${STEPS:-10}"
EP="${EP:-}"
PADDING="${PADDING:-0}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

NNODES="$(wc -l < "${PBS_NODEFILE}")"
JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="${SUBMIT_DIR}/logs/moe-ab-${CONFIG}-ep${EP:-default}-pad${PADDING}-N${NNODES}-${JOBID_SHORT}"
mkdir -p "${LOG_DIR}"

BASE_SHA="$(git rev-parse "${BASELINE_COMMIT}")"
HEAD_SHA="$(git rev-parse "${HEAD_COMMIT}")"
BASE_SHORT="${BASE_SHA:0:9}"
HEAD_SHORT="${HEAD_SHA:0:9}"

WT_BASE="${SUBMIT_DIR}/.claude/worktrees/moe-ab-${JOBID_SHORT}"
WT_BASELINE="${WT_BASE}/baseline"
WT_HEAD="${WT_BASE}/head"
mkdir -p "${WT_BASE}"

echo "==================================================" | tee -a "${LOG_DIR}/run.log"
echo "creating worktrees:" | tee -a "${LOG_DIR}/run.log"
echo "  ${WT_BASELINE} -> ${BASE_SHORT} (baseline=${BASELINE_COMMIT})" | tee -a "${LOG_DIR}/run.log"
echo "  ${WT_HEAD}     -> ${HEAD_SHORT} (head=${HEAD_COMMIT})"     | tee -a "${LOG_DIR}/run.log"
echo "==================================================" | tee -a "${LOG_DIR}/run.log"
git worktree add --detach "${WT_BASELINE}" "${BASE_SHA}" 2>&1 | tee -a "${LOG_DIR}/run.log"
git worktree add --detach "${WT_HEAD}"     "${HEAD_SHA}" 2>&1 | tee -a "${LOG_DIR}/run.log"

# Share .venv + assets/hf via symlinks so we don't re-tar/re-yeet.
ln -sf "${SUBMIT_DIR}/.venv" "${WT_BASELINE}/.venv"
ln -sf "${SUBMIT_DIR}/.venv" "${WT_HEAD}/.venv"
mkdir -p "${WT_BASELINE}/assets" "${WT_HEAD}/assets"
ln -sf "${SUBMIT_DIR}/assets/hf" "${WT_BASELINE}/assets/hf"
ln -sf "${SUBMIT_DIR}/assets/hf" "${WT_HEAD}/assets/hf"

cleanup() {
    local rc=$?
    cd "${SUBMIT_DIR}" 2>/dev/null || true
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "cleanup: removing worktrees" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    rm -f "${WT_BASELINE}/.venv" "${WT_HEAD}/.venv"
    rm -f "${WT_BASELINE}/assets/hf" "${WT_HEAD}/assets/hf"
    git worktree remove --force "${WT_BASELINE}" 2>&1 | tee -a "${LOG_DIR}/run.log" || true
    git worktree remove --force "${WT_HEAD}"     2>&1 | tee -a "${LOG_DIR}/run.log" || true
    rmdir "${WT_BASE}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT

source "${SUBMIT_DIR}/.venv/bin/activate"
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

run_one() {
    local label="$1"
    local wt="$2"
    local commit_short="$3"
    local outlog="${LOG_DIR}/${label}.log"

    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "phase: ${label}  commit: ${commit_short}  cwd: ${wt}" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"

    cd "${wt}"

    local extra=()
    if [[ -n "${EP}" ]]; then
        extra+=(--parallelism.expert-parallel-degree="${EP}")
    fi

    TT_MOE_NORMAL_EQUAL_A2A_PADDING="${PADDING}" \
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module=ezpz.moe \
        --config="${CONFIG}" \
        --compile.no-enable \
        --checkpoint.no-enable \
        --debug.seed=42 \
        --training.steps="${STEPS}" \
        "${extra[@]}" \
        2>&1 | tee "${outlog}"
}

run_one "baseline" "${WT_BASELINE}" "${BASE_SHORT}"
run_one "head"     "${WT_HEAD}"     "${HEAD_SHORT}"

cd "${SUBMIT_DIR}"

extract_metrics() {
    local in="$1" out="$2"
    sed 's/\x1b\[[0-9;]*m//g' "$in" \
        | grep -oE 'step:\s+[0-9]+\s+loss:\s+[0-9.]+\s+grad_norm:\s+[0-9.]+' \
        > "$out"
}
extract_metrics "${LOG_DIR}/baseline.log" "${LOG_DIR}/baseline.metrics"
extract_metrics "${LOG_DIR}/head.log"     "${LOG_DIR}/head.metrics"

diff -u "${LOG_DIR}/baseline.metrics" "${LOG_DIR}/head.metrics" > "${LOG_DIR}/diff.txt"
DIFF_RC=$?

echo "==================================================" | tee -a "${LOG_DIR}/run.log"
echo "config: ${CONFIG}  EP=${EP:-default}  padding=${PADDING}  N=${NNODES}" | tee -a "${LOG_DIR}/run.log"
echo "baseline: ${BASE_SHORT}  head: ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
echo "==================================================" | tee -a "${LOG_DIR}/run.log"
echo "baseline.metrics:" | tee -a "${LOG_DIR}/run.log"
cat "${LOG_DIR}/baseline.metrics" | tee -a "${LOG_DIR}/run.log"
echo "---" | tee -a "${LOG_DIR}/run.log"
echo "head.metrics:" | tee -a "${LOG_DIR}/run.log"
cat "${LOG_DIR}/head.metrics" | tee -a "${LOG_DIR}/run.log"
echo "---" | tee -a "${LOG_DIR}/run.log"
echo "diff:" | tee -a "${LOG_DIR}/run.log"
cat "${LOG_DIR}/diff.txt" | tee -a "${LOG_DIR}/run.log"

if [[ "$DIFF_RC" == "0" ]] && [[ -s "${LOG_DIR}/head.metrics" ]] && [[ -s "${LOG_DIR}/baseline.metrics" ]]; then
    echo "VERDICT: BIT-IDENTICAL" | tee -a "${LOG_DIR}/run.log"
    echo "BIT-IDENTICAL" > "${LOG_DIR}/verdict"
    exit 0
else
    echo "VERDICT: DRIFT (see diff above)" | tee -a "${LOG_DIR}/run.log"
    echo "DRIFT" > "${LOG_DIR}/verdict"
    exit 1
fi
