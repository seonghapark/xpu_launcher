#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Resume 2B 256N backfill — 8533608 walltimed at 8h with 23 ckpts
# remaining (the early-step ckpts step-200 → step-25100). 23 × ~5 min
# = ~2h, fits comfortably in 4h walltime.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="200 400 600 800 1000 1200 1400 1600 1800 2000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 25100" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
