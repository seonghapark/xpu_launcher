#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for xpu launch:
# - single node: ./run_train.sh single -- python train.py --epochs 1
# - multi node : ./run_train.sh multi /path/to/hosts -- python train.py --epochs 1

usage() {
  cat <<'EOF'
Usage:
  run_train.sh single [--dry-run] [-- <train command...>]
  run_train.sh multi [hostfile] [--dry-run] [-- <train command...>]

Examples:
  ./run_train.sh single -- python train.py --epochs 1
  ./run_train.sh multi -- python train.py --epochs 1
  ./run_train.sh multi ./hosts.txt -- python train.py --epochs 1
  NPROC=24 ./run_train.sh multi -- python train.py --epochs 1

Environment overrides:
  XPU_CMD                    (default: xpu)
  SCHEDULER                  (default: auto; auto-detects PBS/SLURM)
  NNODES                     (multi only, default: auto-detect from scheduler or hostfile)
  NPROC                      (default: auto-calculated from NNODES * devices, or inferred by scheduler)
  AUTO_RETRY                 (default: 1, used for multi)
  SPARE_NODES_PERCENTAGE     (multi only, calculate spare as percentage of total available nodes)
  SPARE_NODES                (explicit spare nodes, overrides percentage calculation)
  FAILOVER_PROFILE           (default: auto, used for multi + AUTO_RETRY=1)
  HOST_IP_MAP                (optional JSON file path for deterministic IP->host mapping)

Note: NPROC_PER_NODE is auto-detected from XPU device count (12 devices per Aurora node)

Topology resolution (in order):
  1. Explicit hostfile if provided
  2. PBS_NODEFILE if running inside PBS job
  3. SLURM environment (SLURM_NODELIST, SLURM_JOB_ID) if using SLURM scheduler
  4. User-supplied NNODES/NPROC environment variables
  5. xpu launch auto-detection

Examples:
  # Inside a PBS job (auto-detects from PBS_NODEFILE)
  ./run_train.sh multi -- python train.py

  # With explicit hostfile
  ./run_train.sh multi ./hosts.txt -- python train.py

  # With explicit node count
  NNODES=20 ./run_train.sh multi -- python train.py

  # With hostfile + 20% spare nodes
  SPARE_NODES_PERCENTAGE=20 ./run_train.sh multi ./hosts.txt -- python train.py

  # Explicit: 39 computing nodes + 10 spare
  NNODES=39 SPARE_NODES=10 ./run_train.sh multi -- python train.py
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

DRY_RUN=0
HOSTFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      break
      ;;
    -* )
      echo "error: unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ "$MODE" == "multi" && -z "$HOSTFILE" ]]; then
        HOSTFILE="$1"
        shift
      else
        break
      fi
      ;;
  esac
done

if [[ "$MODE" == "multi" ]]; then
  if [[ -n "$HOSTFILE" && ! -f "$HOSTFILE" ]]; then
    echo "error: hostfile not found: $HOSTFILE" >&2
    exit 1
  fi
fi

if [[ "$MODE" != "single" && "$MODE" != "multi" ]]; then
  echo "error: mode must be 'single' or 'multi'" >&2
  usage
  exit 1
fi

TRAIN_CMD=("python" "train.py")
if [[ "${1:-}" == "--" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    echo "error: expected training command after '--'" >&2
    exit 1
  fi
  TRAIN_CMD=("$@")
fi

XPU_CMD="${XPU_CMD:-xpu}"
# Resolve the interpreter. Order matters: an activated virtualenv first, then a
# .venv beside this script, then the site venv that actually has torch. A bare
# python3 from PATH is the last resort — on Aurora login nodes that is
# /usr/bin/python3, which has no torch. PYTHON_BIN or XPU_VENV override.
XPU_VENV="${XPU_VENV:-/lus/flare/projects/datascience/seonghapark/venv}"
resolve_python_bin() {
  local candidate
  for candidate in \
    "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" \
    "${SCRIPT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/.venv/bin/python" \
    "${XPU_VENV}/bin/python"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done

  command -v python3 2>/dev/null || echo python3
}
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"
SCHEDULER="${SCHEDULER:-auto}"

# Auto-detect number of XPU devices per node
# Each Aurora XPU node has 12 devices
NPROC_PER_NODE="$("$PYTHON_BIN" -c 'import torch; print(torch.xpu.device_count() if hasattr(torch, "xpu") else 12)' 2>/dev/null || echo 12)"

# Silence python import-time warnings on all ranks (XPU_SHOW_WARNINGS=1 re-enables)
if [[ "${XPU_SHOW_WARNINGS:-0}" != "1" ]]; then
  export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
  export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
fi

# Resolve launcher relative to this repo, not the caller's cwd
LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PYTHON_BIN may target a compute-node-only frameworks build; fall back for the
# launcher process itself (the training command still uses PYTHON_BIN as given).
_py_ok() { command -v "$1" >/dev/null 2>&1 && "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; }
LAUNCHER_PY=""
# Last candidate: login-node frameworks python (module may not be loaded)
for cand in "$PYTHON_BIN" "${LAUNCHER_ROOT}/.venv/bin/python" python3 python \
    /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3; do
  if _py_ok "$cand"; then
    LAUNCHER_PY="$cand"
    break
  fi
done

XPU_INVOKE=()
if command -v "$XPU_CMD" >/dev/null 2>&1; then
  XPU_INVOKE=("$XPU_CMD")
elif [[ -n "$LAUNCHER_PY" ]] && [[ -d "${LAUNCHER_ROOT}/src/cli" ]]; then
  export PYTHONPATH="${LAUNCHER_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  XPU_INVOKE=("$LAUNCHER_PY" "-m" "cli")
elif [[ -x "${LAUNCHER_ROOT}/.venv/bin/xpu" ]]; then
  XPU_INVOKE=("${LAUNCHER_ROOT}/.venv/bin/xpu")
else
  echo "error: cannot find xpu launcher (PATH, ${LAUNCHER_ROOT}/src/cli, ${LAUNCHER_ROOT}/.venv/bin/xpu)" >&2
  exit 1
fi

LAUNCH_CMD=("${XPU_INVOKE[@]}" "launch" "--scheduler" "$SCHEDULER")

if [[ "$MODE" == "single" ]]; then
  NPROC="${NPROC:-$NPROC_PER_NODE}"
  LAUNCH_CMD+=("-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")
elif [[ "$MODE" == "multi" ]]; then
  # Calculate total available nodes from hostfile if provided, otherwise let xpu_launch infer
  if [[ -n "$HOSTFILE" ]]; then
    TOTAL_AVAILABLE_NODES="${TOTAL_AVAILABLE_NODES:-$(awk 'NF {print $1}' "$HOSTFILE" | sort -u | wc -l)}"

    # If SPARE_NODES_PERCENTAGE is set, calculate spare nodes as percentage of total available
    if [[ -n "${SPARE_NODES_PERCENTAGE:-}" && -z "${SPARE_NODES:-}" ]]; then
      SPARE_NODES=$((TOTAL_AVAILABLE_NODES * SPARE_NODES_PERCENTAGE / 100))
      [[ $SPARE_NODES -lt 1 ]] && SPARE_NODES=1
      NNODES=$((TOTAL_AVAILABLE_NODES - SPARE_NODES))
    else
      # Default: use all available nodes for computing if not specified
      NNODES="${NNODES:-$TOTAL_AVAILABLE_NODES}"
      SPARE_NODES="${SPARE_NODES:-0}"
    fi

    NPROC="${NPROC:-$((NNODES * NPROC_PER_NODE))}"
    LAUNCH_CMD+=("--hostfile" "$HOSTFILE" "-nh" "$NNODES" "-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")
  else
    # No hostfile: let xpu_launch auto-detect from scheduler (PBS_NODEFILE, SLURM_NODELIST, etc.)
    # Only add topology flags if explicitly set by user
    if [[ -n "${NNODES:-}" ]]; then
      NPROC="${NPROC:-$((NNODES * NPROC_PER_NODE))}"
      LAUNCH_CMD+=("-nh" "$NNODES" "-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")
    elif [[ -n "${NPROC:-}" ]]; then
      # User specified NPROC but not NNODES: just pass it through
      LAUNCH_CMD+=("-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")
    else
      # No topology specified: xpu_launch will infer from scheduler
      LAUNCH_CMD+=("-ppn" "$NPROC_PER_NODE")
    fi
    SPARE_NODES="${SPARE_NODES:-0}"
  fi

  AUTO_RETRY="${AUTO_RETRY:-1}"
  if [[ "$AUTO_RETRY" == "1" && "$SPARE_NODES" -gt 0 ]]; then
    FAILOVER_PROFILE="${FAILOVER_PROFILE:-auto}"
    LAUNCH_CMD+=("--auto-retry" "--spare-nodes" "$SPARE_NODES" "--failover-profile" "$FAILOVER_PROFILE")
  fi

  if [[ -n "${HOST_IP_MAP:-}" ]]; then
    if [[ ! -f "$HOST_IP_MAP" ]]; then
      echo "error: HOST_IP_MAP file not found: $HOST_IP_MAP" >&2
      exit 1
    fi
    LAUNCH_CMD+=("--host-ip-map" "$HOST_IP_MAP")
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  LAUNCH_CMD+=("--dry-run")
fi

LAUNCH_CMD+=("--")
LAUNCH_CMD+=("${TRAIN_CMD[@]}")

printf 'Running: '
printf '%q ' "${LAUNCH_CMD[@]}"
printf '\n'

exec "${LAUNCH_CMD[@]}"
