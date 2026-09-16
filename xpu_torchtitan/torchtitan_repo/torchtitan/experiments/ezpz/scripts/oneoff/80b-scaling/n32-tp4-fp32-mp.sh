#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-tp4-fp32-mp
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# TP=4 + fp32 activations. Prior TP=2 fp32 OOM'd at ~80 GiB. TP=4
# halves activation memory per tile so this should actually fit
# (~40 GiB expected). Same other config as C1 (TP=4 LBS=1 GBS=96).
#
# If this trains clean past step 18 (where C1 NaN'd), the bug IS
# bf16-overflow + can be fixed by promoting activations to fp32 at
# scale. If it still NaN's at step ~18, the bug is elsewhere
# (gradient clipping, normalization, optimizer state, etc.).

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
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-tp4-gbs96-mp-fp32 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.mixed-precision-param=float32
