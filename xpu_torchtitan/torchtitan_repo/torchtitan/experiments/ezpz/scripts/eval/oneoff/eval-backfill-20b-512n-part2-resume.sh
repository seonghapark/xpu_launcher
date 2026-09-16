#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Resume 20B 512N backfill part2 — 8533611 walltimed at 12h with
# 12/18 done (steps 2600-3700 complete). 6 ckpts remain:
# 3800, 3900, 4000, 4100, 4200, 4300. Per-ckpt ~30 min;
# 6 × 30 = 3h, fits in 4h walltime.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="3800 3900 4000 4100 4200 4300" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
