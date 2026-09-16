#!/bin/bash --login
#PBS -A datascience
#PBS -N agpt-80b-conv
#PBS -l walltime=06:00:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe
# PBS submitter wrapper around scripts/run_80b_convergence.sh (the
# in-allocation runner). PBS cannot forward "$@" to a job script, so this
# thin wrapper carries the PBS headers + select= and hands off; all knobs go
# via `qsub -v`.
#
# Usage (Sunspot, 80B head-to-head at the production batch GBS=6144):
#   qsub -l select=64 -l walltime=06:00:00 \
#     -v CONV_OPTIMIZERS="mano sophiag adamw",CONV_STEPS=200 \
#     torchtitan/experiments/ezpz/scripts/submit_80b_convergence.sh
#
# 64 nodes -> dp_degree=186 at TP=4 (the NaN-free <=186 ceiling). See
# run_80b_convergence.sh for the LR recommendations and stable-corner flags.

cd "${PBS_O_WORKDIR:-$(pwd)}"
exec bash torchtitan/experiments/ezpz/scripts/run_80b_convergence.sh
