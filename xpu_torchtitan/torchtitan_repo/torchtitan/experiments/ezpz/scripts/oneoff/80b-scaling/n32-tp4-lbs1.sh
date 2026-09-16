#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-tp4-lbs1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# Test C1: TP=4 with LBS=1. Halves GBS (32×12÷4 = 96 = n=16 baseline)
# and halves dp-shard count to 96. If this trains clean, NaN is either
# (a) GBS-dependent (above 96), or (b) dp_world_size-dependent.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=32 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=4 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-tp4-gbs96 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
