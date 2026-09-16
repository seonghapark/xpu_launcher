#!/usr/bin/bash
# Submit remaining Aurora scaling study jobs that failed due to queue limits.
# Run this after earlier jobs complete to submit the next batch.
#
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/submit_remaining_scaling.sh
#
# Set DRY_RUN=1 to preview without submitting.

set -uo pipefail

TIMESTAMP="${TIMESTAMP:-20260415_110159}"
BENCH_STEPS="${BENCH_STEPS:-20}"
ACCOUNT="${ACCOUNT:-AuroraGPT}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_DIR="torchtitan/experiments/ezpz/scripts"
RUN_SCRIPT="${SCRIPT_DIR}/run_scaling_study_aurora.sh"
SCALING_OUTDIR_BASE="outputs/scaling_study_aurora/${TIMESTAMP}"

# Jobs that still need to be submitted (failed due to queue limits)
# Format: "node_count:group:queue:walltime"
#
# Strategy:
#   1-16 nodes  → capacity queue (2h walltime, 5 queued/user, all configs)
#   32-128 nodes → debug-scaling (1h walltime, split light/heavy)
#   64 light already submitted as 8437382
REMAINING_JOBS=(
    "128:light:debug-scaling:01:00:00"
    "128:heavy:debug-scaling:01:00:00"
)

echo "============================================================"
echo " Submitting remaining scaling study jobs"
echo " Timestamp: ${TIMESTAMP}"
echo " Remaining: ${#REMAINING_JOBS[@]} jobs"
echo "============================================================"
echo ""

submitted=0
failed=0

for entry in "${REMAINING_JOBS[@]}"; do
    IFS=':' read -r nnodes group queue walltime <<< "${entry}"

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
        echo "[DRY] n=${nnodes} group=${group} queue=${queue}"
    else
        tmpfile="$(mktemp /tmp/scaling_n${nnodes}_${group}_XXXXXX.pbs)"
        echo "${pbs_script}" > "${tmpfile}"

        echo -n "n=${nnodes} group=${group} (${queue})... "
        job_id="$(qsub "${tmpfile}" 2>&1)"
        rc=$?
        rm -f "${tmpfile}"

        if ((rc == 0)); then
            echo "OK: ${job_id}"
            ((submitted++))
        else
            echo "FAILED: ${job_id}"
            ((failed++))
        fi
    fi
done

echo ""
echo "Submitted: ${submitted}  Failed: ${failed}  (of ${#REMAINING_JOBS[@]})"
echo "Remove successfully submitted entries from this script, then re-run for any that failed."
