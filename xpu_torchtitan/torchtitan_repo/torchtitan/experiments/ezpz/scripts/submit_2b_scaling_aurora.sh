#!/usr/bin/bash
# 2B-only weak-scaling study at production-matching config across
# {4,8,16,32,64,128,256,512,1024,2048,4096} nodes on Aurora.
#
# Config matches the live 2B 256N+512N v2 production chains:
#   LBS=2, seq_len=8192, compile=on, AC=full, fp32 master
#   GBS scales with N (LBS*NGPUS) — weak scaling
#
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/submit_2b_scaling_aurora.sh
#
# Env overrides:
#   NODE_COUNTS    — default "4 8 16 32 64 128 256 512 1024 2048 4096"
#   BENCH_STEPS    — default 20
#   ACCOUNT        — default AuroraGPT
#   DRY_RUN        — set to 1 to print PBS scripts without submitting

set -uo pipefail

NODE_COUNTS="${NODE_COUNTS:-4 8 16 32 64 128 256 512 1024 2048 4096}"
read -ra NODES_ARRAY <<< "${NODE_COUNTS}"

BENCH_STEPS="${BENCH_STEPS:-20}"
ACCOUNT="${ACCOUNT:-AuroraGPT}"
DRY_RUN="${DRY_RUN:-0}"

WORKDIR="$(pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCALING_OUTDIR_BASE="outputs/scaling_2b_aurora/${TIMESTAMP}"

SCRIPT_DIR="torchtitan/experiments/ezpz/scripts"
RUN_SCRIPT="${SCRIPT_DIR}/run_scaling_study_aurora.sh"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "ERROR: ${RUN_SCRIPT} not found. Run from the repo root."
    exit 1
fi

echo "============================================================"
echo " 2B Scaling Study — Aurora (production-matching LBS=2)"
echo " Timestamp:   ${TIMESTAMP}"
echo " Node counts: ${NODE_COUNTS}"
echo " Steps:       ${BENCH_STEPS}"
echo " Account:     ${ACCOUNT}"
echo " Output:      ${SCALING_OUTDIR_BASE}/"
echo "============================================================"
echo ""

# Queue selection — same logic as submit_scaling_study_aurora.sh but with
# extra tiers for 2048N + 4096N which need prod's medium/large routing.
_queue_for_nodes() {
    local n=$1
    if ((n <= 16)); then
        echo "capacity"
    elif ((n <= 256)); then
        echo "debug-scaling"
    else
        # prod routes to small (256-1024), medium (1025-1919), large (1920+)
        echo "prod"
    fi
}

_walltime_for_nodes() {
    local n=$1
    # 2B-only sweep at BENCH_STEPS=20 needs:
    #   yeet-env (~70s @ 8N -> ~755s @ 4096N per CLAUDE.md table)
    #   DDP init (~5-10 min at large N)
    #   torch.compile (~7-15 min for 2B)
    #   20 training steps (~3 min)
    # Total ~30-45 min worst-case. Pad ~2x for slack.
    if ((n <= 16)); then
        # capacity has 168h max; small env overhead dominated by compile
        echo "01:00:00"
    elif ((n <= 256)); then
        # debug-scaling caps at 1h
        echo "01:00:00"
    elif ((n <= 1024)); then
        # prod-small: 1h is enough; shorter walltime = better backfill priority
        echo "01:30:00"
    elif ((n <= 1919)); then
        # prod-medium
        echo "01:30:00"
    else
        # prod-large: 4096N yeet-env ~13min, total still under 45min; pad ~2x
        echo "02:00:00"
    fi
}

declare -a JOB_IDS JOB_DESCS
total_submitted=0

for nnodes in "${NODES_ARRAY[@]}"; do
    queue="$(_queue_for_nodes "${nnodes}")"
    walltime="$(_walltime_for_nodes "${nnodes}")"
    job_name="2b-scale-n${nnodes}"
    outdir="${SCALING_OUTDIR_BASE}/n${nnodes}"

    # Use SCALING_GROUP=light (which contains agpt_2b only via FILTER_TO_2B)
    # — but the existing light group also includes agpt_20b + moe_2b. Override
    # by setting SCALING_FILTER to agpt_2b which run_scaling_study_aurora.sh
    # currently doesn't support — so wrap with env that picks only the 2B row
    # via FILTER_LABELS.

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

# Run only the 2B config from the scaling study
export BENCH_STEPS=${BENCH_STEPS}
export SCALING_GROUP=light
export FILTER_LABELS=agpt_2b
export SCALING_OUTDIR="${outdir}"
# Production-matching: LBS=2 (vs default 1) — override the row inline
export LBS_OVERRIDE_agpt_2b=2

bash ${RUN_SCRIPT} "\$@"
PBSEOF
    )

    if ((DRY_RUN)); then
        echo "--- [n=${nnodes}, queue=${queue}, wall=${walltime}] ---"
        echo "${pbs_script}"
        echo ""
    else
        tmpfile="$(mktemp /tmp/scale2b_n${nnodes}_XXXXXX.pbs)"
        echo "${pbs_script}" > "${tmpfile}"
        echo -n "Submitting n=${nnodes} (queue=${queue}, wall=${walltime})... "
        job_id="$(qsub "${tmpfile}" 2>&1)"
        rc=$?
        rm -f "${tmpfile}"
        if ((rc == 0)); then
            echo "OK: ${job_id}"
            JOB_IDS+=("${job_id}")
            JOB_DESCS+=("n=${nnodes}")
            ((total_submitted++))
        else
            echo "FAILED: ${job_id}"
        fi
    fi
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
        echo "   ${JOB_DESCS[$i]}  ->  ${JOB_IDS[$i]}"
    done
    echo ""
    echo " Output dir: ${SCALING_OUTDIR_BASE}/"
    echo " Monitor:    qstat -u \$USER | grep 2b-scale"
    echo " Aggregate:  python3 torchtitan/experiments/ezpz/utils/aggregate_scaling.py ${SCALING_OUTDIR_BASE}"
    echo "============================================================"
fi
