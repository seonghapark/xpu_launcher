#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# One-off backfill: re-run 2B step-69900 eval with the new 7-task set
# (existing results.json was moved aside to .results.4task.json).
#
# Delegates to the canonical eval-2b-v2.sh which already handles:
# - frameworks/2025.3.1 module load
# - venvs/aurora/tt-lm-eval activation
# - XPU caching_allocator_warmup monkey-patch
# - HF backend (not vllm — vllm not on XPU)
# - skip-if-results-exist (which is why we moved results.json aside).

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS=69900 \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
