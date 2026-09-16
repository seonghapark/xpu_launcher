#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 2b-scale-n256-failover-verify
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=256
#PBS -q debug-scaling
#PBS -j oe

# Verification harness for n512-failover.sh — same recipe at n=256 where
# we already have known-good scaling data (TPS=5,002 MFU=18.77% from the
# 2026-05-29 bare-launch sweep). If the failover-wrapper path gives a
# similar TPS/MFU at n=256, we trust that whatever it produces at n=512
# is a real data point (vs an artifact of the wrapper itself).
#
# Lives in debug-scaling queue (1h walltime, ≤256 nodes hard cap), so
# we get fast turnaround without burning a prod slot. No spare nodes
# this time because select=256 is already at the queue ceiling; if a
# bad-node hits init the probe will fail cleanly and we can resubmit.
# FAILOVER_MAX_RETRIES=0 reflects that (no spares to swap in).
#
# Once it lands and we read TPS/MFU off the .o log, compare to the
# canonical n=256 LBS=2 row in docs/scaling/agpt-2b.md:
#     n=256 GBS=6,144 TPS=5,002 MFU=18.77% (2026-05-29 sweep)
# If we land near that with the failover path, the 8542015 (n=512)
# probe data is trustworthy.

cd /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz

NHOSTS_TRAIN=256 \
FAILOVER_MAX_RETRIES=0 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=999999 \
CKPT_DIR=outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144-SCALING-PROBE \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_2b_aurora_venv_failover.sh
