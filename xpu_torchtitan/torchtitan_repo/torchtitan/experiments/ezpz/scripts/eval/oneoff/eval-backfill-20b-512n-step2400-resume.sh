#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Last leftover from 20B 512N backfill — 8534628 walltimed at 4h with
# 4/5 done; step-2400 didn't fit. ~30 min walltime needed.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="2400" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
