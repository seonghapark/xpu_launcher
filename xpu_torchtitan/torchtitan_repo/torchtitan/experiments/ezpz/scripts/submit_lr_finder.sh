#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-lr-finder
#PBS -l walltime=06:00:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe
# PBS submitter wrapper around scripts/run_lr_finder.sh.
#
# run_lr_finder.sh is an IN-ALLOCATION runner (it expects module/venv/
# ezpz setup to happen in its own body, then loops over models x
# optimizers calling `ezpz launch ... --lr_finder.enable`). It is not a
# PBS script itself, and PBS cannot forward "$@" to a job script, so this
# thin wrapper exists purely to carry the PBS headers + select= and hand
# off to run_lr_finder.sh. All knobs are passed via `qsub -v`.
#
# Usage (Sunspot, 80B stable-corner sweep at the validated TP=4 corner):
#   qsub -l select=64 -v LRF_MODELS=80b,LRF_OPTIMIZERS="adamw mano" \
#       torchtitan/experiments/ezpz/scripts/submit_lr_finder.sh
#
# The 80B path forces TP=4 + the stable-corner flags (compile OFF,
# AC=full, pure FSDP) inside run_lr_finder.sh so dp_degree stays at the
# NaN-free <=186 ceiling -- a TP=2 sweep would NaN from the dp-degree
# trigger, not the swept LR (see run_lr_finder.sh and
# docs/production/agpt/80b/README.md).
#
# Knobs (all optional, forwarded via the environment to run_lr_finder.sh):
#   LRF_MODELS      space-separated flavors      (default "2b 20b")
#   LRF_OPTIMIZERS  space-separated optimizers   (default "adamw muon sophiag")
#   LRF_TP          tensor-parallel degree (80B) (default 4)
#   LRF_LBS         local batch size             (default 1)
#   LRF_STEPS / LRF_FRACTION / LRF_INIT_LR / LRF_MAX_LR / LRF_TIMEOUT
#
# NOTE: SophiaG and Muon are documented-broken at 80B (bf16 overflow in
# the Hessian / Newton-Schulz at dim=9216 -- see CLAUDE.md); a sweep that
# includes them will likely NaN early at the optimizer level regardless of
# LR. AdamW and Mano are the meaningful 80B sweeps.

cd "${PBS_O_WORKDIR:-$(pwd)}"
exec bash torchtitan/experiments/ezpz/scripts/run_lr_finder.sh
