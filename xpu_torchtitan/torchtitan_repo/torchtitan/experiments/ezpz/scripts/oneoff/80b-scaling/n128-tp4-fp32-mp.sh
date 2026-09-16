#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n128-tp4-fp32-mp
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=134
#PBS -q debug-scaling
#PBS -j oe

# n=128, TP=4, LBS=1, fp32-activations. GBS = 128 * 12/4 * 1 = 384.
# Matches the production-relevant GBS that 8540102 (deterministic)
# NaN'd at. Only submit AFTER 8540167 (n=64 same config GBS=192)
# trains clean — if n=64 fp32-mp NaNs, this one will too and is
# wasted nodes.
#
# 6 spare nodes + FAILOVER_MAX_RETRIES=2 — same overprovisioning
# pattern as the det sweep scripts.
#
# Per CLAUDE.md: NEVER use --debug.deterministic_warn_only.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=128 \
FAILOVER_MAX_RETRIES=2 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=4 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n128-tp4-gbs384-mp-fp32 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.mixed-precision-param=float32
