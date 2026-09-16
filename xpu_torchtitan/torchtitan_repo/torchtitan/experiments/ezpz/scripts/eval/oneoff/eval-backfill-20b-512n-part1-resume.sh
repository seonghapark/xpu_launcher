#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Resume 20B 512N backfill part1 — 8533610 walltimed at 12h with 12/17
# done (steps 100-1400 complete). 5 ckpts remain: 1600/1800/2000/2200/2400.
# Per-ckpt ~30 min; 5 × 30 = 2.5h, fits in 4h walltime.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="1600 1800 2000 2200 2400" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
