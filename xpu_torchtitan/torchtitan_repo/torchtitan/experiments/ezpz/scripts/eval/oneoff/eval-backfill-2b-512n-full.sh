#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill 2B 512N chain with the new 7-task lm-eval set.
# 27 ckpts. Per-ckpt ~7 min, total ~3.2h => fits in 6h walltime.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000 21000 22000 23000 24000 25000 26000 27000 28000 28900 29000 30000" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
