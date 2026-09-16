#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=04:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# One-off backfill: re-run 20B step-4400 eval with the new 7-task set
# (existing results.json was moved aside to .results.4task.json).
#
# Delegates to the canonical eval-20b-v2.sh which already handles the
# frameworks/2025.3.1 + tt-lm-eval venv + XPU monkey-patch + HF backend.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS=4400 \
CKPT_NAME=agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288 \
LABEL=512n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-20b-v2.sh
