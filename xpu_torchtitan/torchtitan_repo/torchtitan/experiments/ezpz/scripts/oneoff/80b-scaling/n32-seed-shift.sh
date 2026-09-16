#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-seed-shift
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# Test D: same config as the failing baseline, just a different
# `--debug.seed` (default 42 → 12345). If NaN moves to a different
# step (or doesn't happen in 20 steps), the failure is data-content
# dependent (a specific token sequence triggers overflow). If NaN
# still hits at step 6, it's not data-dependent.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=32 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=2 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-seed12345 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --debug.seed=12345
