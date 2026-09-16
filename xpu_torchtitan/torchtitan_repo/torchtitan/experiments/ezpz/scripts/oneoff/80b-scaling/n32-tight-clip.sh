#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n32-tight-clip
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=32
#PBS -q debug-scaling
#PBS -j oe

# n=32 baseline config (TP=2 LBS=1 GBS=192, bf16 activations, the
# failing-at-step-6 config) but with tighter gradient clipping:
# max_norm=0.1 vs default 1.0.
#
# Hypothesis: weight drift between optimizer steps is causing the
# bf16 forward overflow at step 6. The TP=4 fp32-activations run
# (8537349) showed pre-clip grad_norms of 21K-79K at steps 4-13,
# which clipping reduces to <=1.0 before the optimizer step — but if
# the optimizer is still moving weights into a regime where the next
# forward overflows, a tighter clip (0.1) would keep weights closer
# to their step-1 state.

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
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n32-gbs192-clip0.1 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.max-norm=0.1
