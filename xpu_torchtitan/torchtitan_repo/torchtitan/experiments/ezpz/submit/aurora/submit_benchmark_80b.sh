#!/bin/bash --login
#PBS -N agpt-80b-benchmark
#PBS -l select=4
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q prod
#PBS -k doe
#PBS -j oe

# Benchmark 80B model configs across parallelism settings.
#
# Default: 4 nodes (48 XPUs). Override via:
#   qsub -l select=8 torchtitan/experiments/ezpz/submit_benchmark_80b.sh
#
# Or pass extra args:
#   qsub -v "BENCH_TP=8,BENCH_PP=1 2" torchtitan/experiments/ezpz/submit_benchmark_80b.sh

cd "${PBS_O_WORKDIR}" || exit 1

exec bash torchtitan/experiments/ezpz/scripts/benchmark_80b.sh "$@"
