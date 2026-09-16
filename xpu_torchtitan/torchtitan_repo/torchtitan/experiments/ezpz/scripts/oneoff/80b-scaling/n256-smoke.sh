#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-scale-n256-smoke
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=256
#PBS -q debug-scaling
#PBS -j oe

# 80B scaling sweep — 256N smoke (20 steps + step-10 sync ckpt save).
# This is the size where attempt 8530891 NaN'd at step 2. Want to
# confirm whether that was deterministic or a one-off; if it NaNs
# again immediately, then we have a real GBS-scaling problem.
# See ../n16-smoke.sh header for goal + config.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=256 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=2 \
LBS=1 \
GAS=1 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
