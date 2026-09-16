#!/bin/bash --login
#PBS -A datascience
#PBS -N bitwise-sync-check
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# Bitwise-equivalence smoke for upstream-sync verification.
#
# Per the root CLAUDE.md project rule:
#   "Non-computation changes (e.g. activation checkpointing, refactoring)
#    must produce identical loss before vs. after with --debug.seed=42
#    and --debug.deterministic."
#
# Runs the same N-step agpt_2b smoke TWICE — once on the current HEAD
# (post-merge), once on a user-specified pre-merge commit — both with
# --debug.seed=42 --debug.deterministic so loss + grad_norm should be
# bit-identical for any pure-refactor / import-rename merge.
#
# Implementation note: each phase runs from an ephemeral `git worktree
# add` checkout under .claude/worktrees/. This avoids `git stash` /
# `git checkout` on the main working tree (which job 12468296 spent its
# full 1h walltime on — the stash -u walk of thousands of untracked
# .venv.tar.gz-* backups + core.* dumps + outputs/ tree never finished).
# Worktrees are sub-second to create and don't touch the main tree at
# all.
#
# Usage (from a login node, with this script as the qsub argument):
#
#   qsub -A datascience -q workq -l select=2 -l walltime=01:00:00 \
#     -l filesystems=flare:home \
#     -v PRE_MERGE_COMMIT=8b610e911,STEPS=20 \
#     torchtitan/experiments/ezpz/scripts/bitwise_sync_check.sh
#
# Env knobs:
#   PRE_MERGE_COMMIT  required — sha to compare HEAD against (defaults
#                      to HEAD's first parent if omitted)
#   STEPS             default: 20  (cheap; enough to surface drift)
#   MODEL             default: 2b
#   SEED              default: 42  (matches CLAUDE.md convention)
#
# Output: logs/bitwise-sync-check-${PBS_JOBID}/
#   head.log         post-merge run output
#   pre.log          pre-merge run output
#   head.metrics     extracted (step, loss, grad_norm) from head.log
#   pre.metrics      same from pre.log
#   diff.txt         diff between the two .metrics files
#   verdict          "IDENTICAL" if diff is empty, "DRIFT" otherwise

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

HEAD_COMMIT="${HEAD_COMMIT:-$(git rev-parse HEAD)}"
PRE_MERGE_COMMIT="${PRE_MERGE_COMMIT:-$(git rev-parse HEAD^1 2>/dev/null)}"
if [[ -z "$PRE_MERGE_COMMIT" ]] || ! git rev-parse "$PRE_MERGE_COMMIT" >/dev/null 2>&1; then
    echo "FATAL: PRE_MERGE_COMMIT='$PRE_MERGE_COMMIT' is not a valid git ref"
    exit 1
fi
PRE_MERGE_SHA="$(git rev-parse "$PRE_MERGE_COMMIT")"
HEAD_SHORT="${HEAD_COMMIT:0:9}"
PRE_SHORT="${PRE_MERGE_SHA:0:9}"

STEPS="${STEPS:-20}"
MODEL="${MODEL:-2b}"
SEED="${SEED:-42}"
NNODES="$(wc -l < "${PBS_NODEFILE}")"
JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="${SUBMIT_DIR}/logs/bitwise-sync-check-${JOBID_SHORT}"
mkdir -p "${LOG_DIR}"

# Ephemeral worktrees — one per commit. Detached HEAD so we don't
# need to mint or clean up branch refs. Worktrees live under
# .claude/worktrees/ which is already gitignored by convention here.
WT_BASE="${SUBMIT_DIR}/.claude/worktrees/bitwise-sync-${JOBID_SHORT}"
WT_HEAD="${WT_BASE}/head"
WT_PRE="${WT_BASE}/pre"
mkdir -p "${WT_BASE}"

echo "==================================================" | tee -a "${LOG_DIR}/run.log"
echo "creating worktrees:" | tee -a "${LOG_DIR}/run.log"
echo "  ${WT_HEAD} -> ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
echo "  ${WT_PRE}  -> ${PRE_SHORT}" | tee -a "${LOG_DIR}/run.log"
echo "==================================================" | tee -a "${LOG_DIR}/run.log"
git worktree add --detach "${WT_HEAD}" "${HEAD_COMMIT}" 2>&1 | tee -a "${LOG_DIR}/run.log"
git worktree add --detach "${WT_PRE}" "${PRE_MERGE_SHA}" 2>&1 | tee -a "${LOG_DIR}/run.log"

# Share the .venv from the main repo so we don't re-tar / re-yeet
# (8.6 GB venv). Symlink instead of copy.
ln -sf "${SUBMIT_DIR}/.venv" "${WT_HEAD}/.venv"
ln -sf "${SUBMIT_DIR}/.venv" "${WT_PRE}/.venv"

# `assets/hf/*` is gitignored (HF tokenizer dirs are too big to track),
# so each worktree has an empty `assets/hf/` but training needs
# `./assets/hf/gemma-7b/tokenizer.model` (relative path resolved against
# the worktree CWD). Symlink the main repo's assets/hf into each
# worktree — same trick the other long-lived worktrees use.
mkdir -p "${WT_HEAD}/assets" "${WT_PRE}/assets"
ln -sf "${SUBMIT_DIR}/assets/hf" "${WT_HEAD}/assets/hf"
ln -sf "${SUBMIT_DIR}/assets/hf" "${WT_PRE}/assets/hf"

cleanup() {
    local rc=$?
    cd "${SUBMIT_DIR}" 2>/dev/null || true
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "cleanup: removing worktrees" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    # Drop the .venv + assets/hf symlinks first so `worktree remove`
    # doesn't try to crawl into them (slow and pointless).
    rm -f "${WT_HEAD}/.venv" "${WT_PRE}/.venv"
    rm -f "${WT_HEAD}/assets/hf" "${WT_PRE}/assets/hf"
    git worktree remove --force "${WT_HEAD}" 2>&1 | tee -a "${LOG_DIR}/run.log" || true
    git worktree remove --force "${WT_PRE}" 2>&1 | tee -a "${LOG_DIR}/run.log" || true
    rmdir "${WT_BASE}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT

source "${SUBMIT_DIR}/.venv/bin/activate"
python3 -c "import torch; print('torch', torch.__version__)" \
    | tee -a "${LOG_DIR}/run.log"

# Data cache shared between both phases — index built once on phase 1,
# phase 2 hits the warm cache. Keeps both phases hermetic w.r.t. each
# other (same data ordering, same shuffle indices).
SHARED_CACHE="${SUBMIT_DIR}/checkpoints/bitwise-sync-check-${JOBID_SHORT}/.cache/books/index-cache"

run_one() {
    local label="$1"
    local wt="$2"
    local commit="$3"
    local outlog="${LOG_DIR}/${label}.log"

    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "phase: ${label}  commit: ${commit:0:9}  cwd: ${wt}" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"

    cd "${wt}"

    local NGPUS_LOCAL="${NHOSTS:-$NNODES}"
    local GBS=$(( NGPUS_LOCAL * 12 ))

    # Memory-fit overrides for the bitwise smoke:
    #   - `agpt_${MODEL}_chunkedce`         vs `agpt_${MODEL}`: chunks the
    #     vocab=256k logit slice (~16 GB at LBS=2) into 8 pieces (~2 GB
    #     each). Mathematically equivalent — sum of chunked CE == full CE.
    #   - `activation-checkpoint:full` vs the default policy on
    #     `ezpz_agpt_2b`: trades attention activation memory for recompute.
    #     Numerically identical compute. Without this, attention activations
    #     at SEQ_LEN=8192 + LBS=1 with `--debug.deterministic` (which uses
    #     more workspace than the default kernels) OOM in SDPA forward at
    #     the first training step (jobs 12468306, 12468308). 57th sync
    #     (PR #3674): AC is now a positional tyro subcommand, passed last.
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module=ezpz.agpt \
        --config="agpt_${MODEL}_chunkedce" \
        --compile.no-enable \
        --checkpoint.no-enable \
        --dataloader.dataset=blendcorpus \
        --dataloader.dataset-path="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt" \
        --dataloader.data-cache-path="${SHARED_CACHE}" \
        --debug.seed="${SEED}" \
        --debug.deterministic \
        --training.local-batch-size=1 \
        --training.global-batch-size="${GBS}" \
        --training.seq-len=8192 \
        --training.steps="${STEPS}" \
        --optimizer=sophiag \
        --optimizer.lr=2.28e-5 \
        activation-checkpoint:full \
        2>&1 | tee "${outlog}"
}

run_one "head" "${WT_HEAD}" "${HEAD_COMMIT}"
run_one "pre" "${WT_PRE}" "${PRE_MERGE_SHA}"

cd "${SUBMIT_DIR}"

extract_metrics() {
    local in="$1" out="$2"
    sed 's/\x1b\[[0-9;]*m//g' "$in" \
        | grep -oE 'step:\s+[0-9]+\s+loss:\s+[0-9.]+\s+grad_norm:\s+[0-9.]+' \
        > "$out"
}
extract_metrics "${LOG_DIR}/head.log" "${LOG_DIR}/head.metrics"
extract_metrics "${LOG_DIR}/pre.log"  "${LOG_DIR}/pre.metrics"

diff -u "${LOG_DIR}/pre.metrics" "${LOG_DIR}/head.metrics" > "${LOG_DIR}/diff.txt"
DIFF_RC=$?

if [[ "$DIFF_RC" == "0" ]] && [[ -s "${LOG_DIR}/head.metrics" ]] && [[ -s "${LOG_DIR}/pre.metrics" ]]; then
    echo "IDENTICAL" > "${LOG_DIR}/verdict"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "VERDICT: IDENTICAL — loss + grad_norm match bit-for-bit" \
        | tee -a "${LOG_DIR}/run.log"
    echo "    pre:  ${PRE_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    head: ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    exit 0
else
    echo "DRIFT" > "${LOG_DIR}/verdict"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    echo "VERDICT: DRIFT — see ${LOG_DIR}/diff.txt" | tee -a "${LOG_DIR}/run.log"
    echo "    pre:  ${PRE_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    head: ${HEAD_SHORT}" | tee -a "${LOG_DIR}/run.log"
    echo "    diff (first 40 lines):" | tee -a "${LOG_DIR}/run.log"
    head -40 "${LOG_DIR}/diff.txt" | tee -a "${LOG_DIR}/run.log"
    echo "==================================================" | tee -a "${LOG_DIR}/run.log"
    exit 1
fi
