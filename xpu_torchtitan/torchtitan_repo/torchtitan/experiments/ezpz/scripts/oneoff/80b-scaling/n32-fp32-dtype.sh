#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-fp32-dtype
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# Test A: forces full fp32 (vs current fp32-master + bf16 activations).
# Hypothesis: the n=32 step-6 NaN comes from a bf16 overflow in
# activations/gradients. Forcing fp32 throughout would eliminate
# that — at significant throughput cost, but as a diagnostic.

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
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-fp32 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.dtype=float32
