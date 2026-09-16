#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill 20B 512N chain (part 1 of 2) with the new 7-task lm-eval set.
# 18 ckpts (step-100 → step-2400). Per-ckpt ~45 min, total ~13h
# (slight overflow — script's skip-if-results-exist handles restart).
# step-4400 excluded (re-run as smoke earlier today via job 8533570).

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400" \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
