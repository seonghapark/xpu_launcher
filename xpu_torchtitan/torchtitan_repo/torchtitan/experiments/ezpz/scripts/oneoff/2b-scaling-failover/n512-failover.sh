#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N 2b-scale-n512-failover
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -l select=515
#PBS -q prod
#PBS -j oe

# NOTE: 512N is past the debug-scaling cap (256 nodes max). Submitted
# via `prod` (routes to `small`) which means 12h walltime min — we'll
# only use ~30min of it. The probe writes to a throwaway CKPT_DIR with
# CKPT_INTERVAL=999999 so it never persists; once TRAINING_STEPS=20 are
# done the job exits and the rest of the walltime is wasted (acceptable
# for a one-shot scaling data point).

# 2B scaling at n=512 via the PRODUCTION failover path (not bare ezpz
# launch). Bare-launch path keeps hitting `set_determinism std::bad_alloc`
# at 6,144 ranks — see project_1024n_init_crash. Production failover
# wrapper has a preflight + retry path that ducks this (the 2B 512N
# production chain trained successfully through it).
#
# 20 training steps in a throwaway CKPT_DIR (so it doesn't pollute the
# canonical 512N chain at /flare/.../agpt-2b-v2/.../n512-gbs12288/).
# Overprovisioned (select=515, +3 spares) with FAILOVER_MAX_RETRIES=2
# so a bad-node hit at init swaps in a spare instead of killing the
# entire benchmark.
#
# Per CLAUDE.md / scaling docs convention: GBS = NGPUS × LBS × GAS / TP
# At n=512, TP=1, LBS=2, GAS=1: GBS = 6,144 × 2 / 1 = 12,288 (matches
# production 2B 512N chain).

cd /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz

NHOSTS_TRAIN=512 \
FAILOVER_MAX_RETRIES=2 \
TRAINING_STEPS=20 \
CKPT_INTERVAL=999999 \
CKPT_DIR=outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288-SCALING-PROBE \
bash torchtitan/experiments/ezpz/scripts/submit_agpt_2b_aurora_venv_failover.sh
