#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n128-det
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=134
#PBS -q debug-scaling
#PBS -j oe

# n=128 (GBS=768) + --debug.deterministic. See n64-det.sh header.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=128 \
FAILOVER_MAX_RETRIES=2 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=2 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n128-gbs768-det \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --debug.seed=42 \
    --debug.deterministic
