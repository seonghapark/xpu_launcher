#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-det-tight-clip
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# n=32 baseline + DETERMINISTIC mode + --training.max-norm=0.1.
# Paired with n32-det.sh (same config, default max_norm=1.0). Together
# they isolate the effect of the clip-norm change at bit-level — both
# use seed=42 and --debug.deterministic so anything that differs
# step-by-step is *purely* the clip-norm changing the math.
#
# Per CLAUDE.md: NEVER use --debug.deterministic_warn_only.

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
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-det-clip0.1 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --debug.seed=42 \
    --debug.deterministic \
    --training.max-norm=0.1
