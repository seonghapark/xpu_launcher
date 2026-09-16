#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill 20B 512N chain (part 2 of 2) with the new 7-task lm-eval set.
# 17 ckpts (step-2600 → step-4300). step-4400 excluded.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 3600 3700 3800 3900 4000 4100 4200 4300" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
