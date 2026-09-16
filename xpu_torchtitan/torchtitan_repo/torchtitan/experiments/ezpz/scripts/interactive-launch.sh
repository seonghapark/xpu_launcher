#!/bin/bash --login
# Interactive launch script — runs inside an existing PBS allocation
# Usage: ssh <head_node> "PBS_JOBID=... PBS_NODEFILE=... bash /path/to/.ezpz-interactive-launch.sh"

: "${PBS_JOBID:?PBS_JOBID must be set}"
: "${PBS_NODEFILE:?PBS_NODEFILE must be set}"

# ALCF compute nodes need proxy for internet access
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

export PBS_O_WORKDIR="${PBS_O_WORKDIR:-/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz}"
cd "${PBS_O_WORKDIR}"

LOGFILE="${PBS_O_WORKDIR}/interactive-launch-$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOGFILE}" >&2

{
    # set +u needed: lmod/ezpz reference unset vars
    set +u
    source <(curl -fsSL https://bit.ly/ezpz-utils)
    ezpz_setup_env

    if ! command -v ezpz >/dev/null; then
        uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
    fi
    set -u

    ezpz launch "$@"
} 2>&1 | tee "${LOGFILE}"
