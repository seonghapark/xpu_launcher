#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 80b-n64-tp4-fp32-mp
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=68
#PBS -q debug-scaling
#PBS -j oe

# n=64, TP=4, LBS=1, fp32-activations. GBS = 64 * 12/4 * 1 = 192.
# Matches the n=32-det GBS=192 that trained clean — first probe of
# whether the validated TP=4 fp32-acts fix survives doubling node
# count at fixed GBS. If clean, follow up at n=128 LBS=1 (GBS=384)
# to confirm production scaling. If NaN, fp32-acts isn't a real fix
# either and the bug is elsewhere (see counter-evidence section of
# 20260611-80b-n32-nan-diagnosis.md).
#
# 4 spare nodes + FAILOVER_MAX_RETRIES=2 for the same reason as
# n64-det.sh — bad-node hits at this scale are common and the run is
# expensive enough to be worth the small overprovision.
#
# Expected memory: ~42 GiB per tile (extrapolated from n=32 TP=4
# fp32-mp run 8537349). Well under the 64 GiB cap.
#
# Per CLAUDE.md: NEVER use --debug.deterministic_warn_only.

cd /flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz

NHOSTS_TRAIN=64 \
FAILOVER_MAX_RETRIES=2 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=10 \
OPTIMIZER=adamw \
LR=1e-6 \
TP=4 \
LBS=1 \
GAS=1 \
CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n64-tp4-gbs192-mp-fp32 \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh \
    --training.mixed-precision-param=float32
