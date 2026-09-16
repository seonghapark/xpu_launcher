#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-scale-n16-smoke
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=16
#PBS -q debug-scaling
#PBS -j oe

# 80B scaling sweep — 16N smoke (20 steps + step-10 sync ckpt save).
# Goal: surface any N-specific breakage at this scale before promoting
# to production. Uses validated 4N config (AdamW LR=1e-6, TP=2,
# AC=full, compile=OFF, fp32-master, sync ckpt).
#
# Writes to its own ckpt dir (agpt-80b-adamw-olmo-mix-1124-n16-gbs96)
# to avoid stepping on production trajectories.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=16 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=2 \
LBS=1 \
GAS=1 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
