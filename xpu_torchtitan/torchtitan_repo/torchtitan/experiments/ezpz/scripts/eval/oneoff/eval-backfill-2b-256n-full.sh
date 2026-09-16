#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=08:00:00
#PBS -l filesystems=home:flare
#PBS -q capacity
#PBS -l select=1
#PBS -j oe

# Backfill 2B 256N chain with the new 7-task lm-eval set.
# 54 ckpts (step-69900 excluded — re-run as smoke earlier today via
# job 8533569, baseline numbers in step-69900/results/results.json).
#
# Per-ckpt time: ~7 min (HellaSwag dominates).
# 54 * 7 = ~6.5h => fits in 8h walltime.
# eval-2b-v2.sh skip-if-results-exist will skip any ckpt already done
# if this job restarts.

cd "${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"

STEPS="200 400 600 800 1000 1200 1400 1600 1800 2000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 25100 36000 37000 38000 38800 39000 40000 41000 42000 42500 43000 44000 45000 45500 46000 47000 48000 49000 49500 50000 51000 51700 52000 54000 56000 58000 60000 62000 64000 66000 68000 69000" \
CKPT_NAME=agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144 \
LABEL=256n \
TASKS="hellaswag,arc_easy,arc_challenge,winogrande,piqa,openbookqa,boolq" \
bash torchtitan/experiments/ezpz/scripts/eval/eval-2b-v2.sh
