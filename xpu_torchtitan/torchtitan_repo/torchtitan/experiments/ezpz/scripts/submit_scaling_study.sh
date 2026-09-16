#!/usr/bin/bash
# Submit PBS jobs for a multi-node scaling study on Sunspot.
# Submits one job per node count in NODE_COUNTS, each running
# run_scaling_study.sh inside the allocation.
#
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/submit_scaling_study.sh
#
# Environment variables:
#   NODE_COUNTS   — space-separated node counts (default: "2 4 8 16 32 64 128")
#   WALLTIME      — per-job walltime (default: "02:00:00")
#   ACCOUNT       — PBS account (default: "datascience")
#   QUEUE         — PBS queue (default: "workq")
#   FILESYSTEMS   — PBS filesystems (default: "home:flare")
#   BENCH_STEPS   — forwarded to run_scaling_study.sh (default: 20)
#   DRY_RUN       — set to 1 to print PBS scripts without submitting

set -uo pipefail

NODE_COUNTS="${NODE_COUNTS:-2 4 8 16 32 64 128}"
read -ra NODES_ARRAY <<< "${NODE_COUNTS}"

WALLTIME="${WALLTIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-datascience}"
QUEUE="${QUEUE:-workq}"
FILESYSTEMS="${FILESYSTEMS:-home:flare}"
BENCH_STEPS="${BENCH_STEPS:-20}"
DRY_RUN="${DRY_RUN:-0}"

WORKDIR="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCALING_OUTDIR_BASE="outputs/scaling_study/${TIMESTAMP}"

echo "============================================================"
echo " Scaling Study Job Submission — ${TIMESTAMP}"
echo " Node counts: ${NODE_COUNTS}"
echo " Walltime: ${WALLTIME}"
echo " Account: ${ACCOUNT}"
echo " Queue: ${QUEUE}"
echo " Steps: ${BENCH_STEPS}"
echo " Output: ${SCALING_OUTDIR_BASE}/"
echo "============================================================"
echo ""

declare -a JOB_IDS

for nnodes in "${NODES_ARRAY[@]}"; do
    job_name="scaling-n${nnodes}"
    outdir="${SCALING_OUTDIR_BASE}/n${nnodes}"

    pbs_script=$(cat <<PBSEOF
#!/bin/bash --login
#PBS -N ${job_name}
#PBS -l select=${nnodes}
#PBS -l walltime=${WALLTIME}
#PBS -l filesystems=${FILESYSTEMS}
#PBS -A ${ACCOUNT}
#PBS -q ${QUEUE}
#PBS -k doe
#PBS -j oe

cd "\${PBS_O_WORKDIR}"

export BENCH_STEPS=${BENCH_STEPS}
export SCALING_OUTDIR="${outdir}"

bash torchtitan/experiments/ezpz/scripts/run_scaling_study.sh "\$@"
PBSEOF
    )

    if ((DRY_RUN)); then
        echo "--- [n=${nnodes}] PBS script (DRY RUN) ---"
        echo "${pbs_script}"
        echo ""
    else
        # Write temp PBS script and submit
        tmpfile="$(mktemp /tmp/scaling_n${nnodes}_XXXXXX.pbs)"
        echo "${pbs_script}" > "${tmpfile}"

        echo -n "Submitting n=${nnodes} (${job_name})... "
        job_id="$(qsub "${tmpfile}" 2>&1)"
        rc=$?
        rm -f "${tmpfile}"

        if ((rc == 0)); then
            echo "OK: ${job_id}"
            JOB_IDS+=("${job_id}")
        else
            echo "FAILED: ${job_id}"
        fi
    fi
done

echo ""

if ((DRY_RUN)); then
    echo "Dry run complete. Set DRY_RUN=0 to submit."
else
    echo "============================================================"
    echo " Submitted ${#JOB_IDS[@]} jobs:"
    for ((i = 0; i < ${#NODES_ARRAY[@]}; i++)); do
        if ((i < ${#JOB_IDS[@]})); then
            echo "   n=${NODES_ARRAY[$i]}  →  ${JOB_IDS[$i]}"
        fi
    done
    echo ""
    echo " Output dir: ${SCALING_OUTDIR_BASE}/"
    echo " Monitor:    qstat -u \$USER"
    echo " Aggregate:  python3 torchtitan/experiments/ezpz/utils/aggregate_scaling.py ${SCALING_OUTDIR_BASE}"
    echo "============================================================"
fi
