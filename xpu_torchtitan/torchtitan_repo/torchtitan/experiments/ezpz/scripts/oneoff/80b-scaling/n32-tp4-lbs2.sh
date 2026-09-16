#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-tp4-lbs2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# Test C2: TP=4 with LBS=2. Preserves GBS=192 (matches baseline) but
# halves dp-shard count + doubles per-replica step batch. Isolates
# TP=2 from TP=4 at the same effective optimizer batch.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=32 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=4 \
LBS=2 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-tp4-lbs2-gbs192 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
