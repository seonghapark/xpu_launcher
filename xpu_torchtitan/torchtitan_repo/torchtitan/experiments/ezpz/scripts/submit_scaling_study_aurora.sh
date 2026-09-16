#!/usr/bin/bash
# Submit PBS jobs for an Aurora scaling study.
# Submits parallel jobs across node counts, split by config group
# to respect queue walltime constraints.
#
# Queue mapping:
#   1–128 nodes   → debug-scaling (1h walltime, 2 jobs per N: light + heavy)
#   256–1024 nodes → prod (6h walltime, 1 combined job per N)
#
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/submit_scaling_study_aurora.sh
#
# Environment variables:
#   NODE_COUNTS     — space-separated node counts
#                     (default: "1 2 4 8 16 32 64 128 256 512 1024")
#   BENCH_STEPS     — forwarded to run_scaling_study_aurora.sh (default: 20)
#   ACCOUNT         — PBS account (default: AuroraGPT)
#   DRY_RUN         — set to 1 to print PBS scripts without submitting

set -uo pipefail

NODE_COUNTS="${NODE_COUNTS:-1 2 4 8 16 32 64 128 256 512 1024}"
read -ra NODES_ARRAY <<< "${NODE_COUNTS}"

BENCH_STEPS="${BENCH_STEPS:-20}"
ACCOUNT="${ACCOUNT:-AuroraGPT}"
DRY_RUN="${DRY_RUN:-0}"

WORKDIR="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCALING_OUTDIR_BASE="outputs/scaling_study_aurora/${TIMESTAMP}"

SCRIPT_DIR="torchtitan/experiments/ezpz/scripts"
RUN_SCRIPT="${SCRIPT_DIR}/run_scaling_study_aurora.sh"

# Verify the run script exists
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: ${RUN_SCRIPT} not found. Run from the repo root."
    exit 1
fi

echo "============================================================"
echo " Aurora Scaling Study — Job Submission"
echo " Timestamp:   ${TIMESTAMP}"
echo " Node counts: ${NODE_COUNTS}"
echo " Steps:       ${BENCH_STEPS}"
echo " Account:     ${ACCOUNT}"
echo " Output:      ${SCALING_OUTDIR_BASE}/"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Queue selection logic
# ---------------------------------------------------------------------------
_queue_for_nodes() {
    local n=$1
    if ((n <= 16)); then
        # capacity: 1-16 nodes, 168h walltime, 5 queued per user
        echo "capacity"
    elif ((n <= 256)); then
        # debug-scaling: 1-256 nodes, 1h walltime
        echo "debug-scaling"
    else
        # prod routes to small (256-1024), medium (1025-1919), large (1920+)
        echo "prod"
    fi
}

_walltime_for_nodes() {
    local n=$1
    if ((n <= 16)); then
        # capacity has 168h max — use 2h for plenty of margin
        echo "02:00:00"
    elif ((n <= 256)); then
        echo "01:00:00"
    else
        echo "06:00:00"
    fi
}

_groups_for_nodes() {
    local n=$1
    if ((n <= 16)); then
        # capacity has 2h walltime — run all configs in one job
        echo "all"
    elif ((n <= 256)); then
        # debug-scaling: only 1h — split into two jobs
        echo "light heavy"
    else
        # prod: plenty of walltime — run all configs in one job
        echo "all"
    fi
}

# ---------------------------------------------------------------------------
# Submit jobs
# ---------------------------------------------------------------------------
declare -a JOB_IDS
declare -a JOB_DESCS
total_submitted=0

for nnodes in "${NODES_ARRAY[@]}"; do
    queue="$(_queue_for_nodes "${nnodes}")"
    walltime="$(_walltime_for_nodes "${nnodes}")"
    groups="$(_groups_for_nodes "${nnodes}")"

    for group in ${groups}; do
        job_name="scaling-n${nnodes}-${group}"
        outdir="${SCALING_OUTDIR_BASE}/n${nnodes}"

        pbs_script=$(cat <<PBSEOF
#!/bin/bash --login
#PBS -N ${job_name}
#PBS -l select=${nnodes}
#PBS -l walltime=${walltime}
#PBS -l filesystems=home:flare
#PBS -A ${ACCOUNT}
#PBS -q ${queue}
#PBS -k doe
#PBS -j oe

cd "\${PBS_O_WORKDIR}"

export BENCH_STEPS=${BENCH_STEPS}
export SCALING_GROUP=${group}
export SCALING_OUTDIR="${outdir}"

bash ${RUN_SCRIPT} "\$@"
PBSEOF
        )

        if ((DRY_RUN)); then
            echo "--- [n=${nnodes}, group=${group}] PBS script (DRY RUN) ---"
            echo "    queue=${queue}  walltime=${walltime}"
            echo "${pbs_script}"
            echo ""
        else
            tmpfile="$(mktemp /tmp/scaling_n${nnodes}_${group}_XXXXXX.pbs)"
            echo "${pbs_script}" > "${tmpfile}"

            echo -n "Submitting n=${nnodes} group=${group} (queue=${queue})... "
            job_id="$(qsub "${tmpfile}" 2>&1)"
            rc=$?
            rm -f "${tmpfile}"

            if ((rc == 0)); then
                echo "OK: ${job_id}"
                JOB_IDS+=("${job_id}")
                JOB_DESCS+=("n=${nnodes} ${group}")
                ((total_submitted++))
            else
                echo "FAILED: ${job_id}"
            fi
        fi
    done
done

echo ""

if ((DRY_RUN)); then
    echo "============================================================"
    echo " Dry run complete — ${#NODES_ARRAY[@]} node counts"
    echo " Set DRY_RUN=0 to submit."
    echo "============================================================"
else
    echo "============================================================"
    echo " Submitted ${total_submitted} jobs:"
    for ((i = 0; i < ${#JOB_IDS[@]}; i++)); do
        echo "   ${JOB_DESCS[$i]}  →  ${JOB_IDS[$i]}"
    done
    echo ""
    echo " Output dir: ${SCALING_OUTDIR_BASE}/"
    echo " Monitor:    qstat -u \$USER"
    echo " Aggregate:  python3 torchtitan/experiments/ezpz/utils/aggregate_scaling.py ${SCALING_OUTDIR_BASE}"
    echo "============================================================"
fi
