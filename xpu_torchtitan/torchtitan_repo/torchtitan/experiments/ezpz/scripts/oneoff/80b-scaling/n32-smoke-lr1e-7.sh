#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-scale-n32-smoke-lr1e-7
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# 80B scaling sweep — 32N smoke RETRY with LR=1e-7 (vs prior 1e-6).
# n=32 LR=1e-6 (8536249) NaN'd at step 2 with grad_norm=inf, then full
# NaN by step 7. Hypothesis: large-batch instability — LR too high
# relative to per-rank batch size at GBS=192. LR=1e-7 (10× lower)
# should ride below the instability threshold.
#
# Uses a different ckpt dir suffix to keep this distinct from the prior
# 1e-6 attempt's ckpts.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=32 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-7 \
TP=2 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-lr1e-7 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
