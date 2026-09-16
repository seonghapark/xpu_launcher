#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n256-det
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=266
#PBS -q debug-scaling
#PBS -j oe

# n=256 (GBS=1536) + --debug.deterministic. This is the production
# scale — if it trains clean past step 6, --debug.deterministic
# unblocks the 80B 256N production chain (currently stuck since
# 2026-05-11 due to this NaN). See n64-det.sh header.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=256 \
FAILOVER_MAX_RETRIES=2 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=2 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n256-gbs1536-det \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --debug.seed=42 \
    --debug.deterministic
