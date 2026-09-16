#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-fp32-mixed-precision
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# Test A (corrected): forces fp32 ACTIVATIONS via
# --training.mixed-precision-param=float32. Prior attempt
# (n32-fp32-dtype.sh, job 8536657) only set --training.dtype=float32
# which controls the master weight dtype (already fp32 in v2), not the
# mixed-precision activation dtype that defaults to bfloat16. Memory
# stayed at 41.68 GiB confirming activations weren't promoted to fp32.
#
# This run actually forces fp32 throughout — expect 2× memory bump
# (~80 GiB peak) and slower throughput, but if NaN goes away, the
# bug is a bf16 overflow in activations/grads.

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
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-mp-fp32 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.mixed-precision-param=float32
