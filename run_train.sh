#!/usr/bin/env bash
set -euo pipefail

# Example wrapper for xpu launch:
# - single node: ./run_train.sh single -- python train.py --epochs 1
# - multi node : ./run_train.sh multi /path/to/hosts -- python train.py --epochs 1

usage() {
  cat <<'EOF'
Usage:
  run_train.sh single [--dry-run] [-- <train command...>]
  run_train.sh multi <hostfile> [--dry-run] [-- <train command...>]

Examples:
  ./run_train.sh single -- python train.py --epochs 1
  ./run_train.sh multi ./hosts.txt -- python train.py --epochs 1
  NPROC=24 ./run_train.sh multi ./hosts.txt -- python train.py --epochs 1

Environment overrides:
  XPU_CMD                    (default: xpu)
  SCHEDULER                  (default: auto)
  NNODES                     (multi only, default: derived from hostfile unique lines)
  NPROC                      (default: NNODES * detected_devices for multi, else detected_devices)
  AUTO_RETRY                 (default: 1, used for multi)
  SPARE_NODES_PERCENTAGE     (multi only, calculate spare as percentage of total available nodes)
  SPARE_NODES                (explicit spare nodes, overrides percentage calculation)
  FAILOVER_PROFILE           (default: auto, used for multi + AUTO_RETRY=1)
  HOST_IP_MAP                (optional JSON file path for deterministic IP->host mapping)

Note: NPROC_PER_NODE is auto-detected from XPU device count (12 devices per Aurora node)

Examples:
  # Use all 49 nodes as computing nodes
  ./run_train.sh multi hostfile.txt -- python train.py

  # Use 49 nodes with 20% as spare (39 computing + 10 spare)
  SPARE_NODES_PERCENTAGE=20 ./run_train.sh multi hostfile.txt -- python train.py

  # Explicit: 39 computing nodes + 10 spare
  NNODES=39 SPARE_NODES=10 ./run_train.sh multi hostfile.txt -- python train.py
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
  if [[ -z "$HOSTFILE" ]]; then
    if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
      HOSTFILE="$(mktemp /tmp/xpu_hosts.XXXXXX)"
      awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$HOSTFILE"
      echo "[INFO] multi mode: derived hostfile from PBS_NODEFILE ($(wc -l < "$HOSTFILE") nodes)" >&2
    else
      echo "error: multi mode requires <hostfile> (or run inside a PBS job with PBS_NODEFILE)" >&2
      exit 1
    fi
  fi
  if [[ ! -f "$HOSTFILE" ]]; then
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
PYTHON_BIN="${PYTHON_BIN:-/lus/flare/projects/datascience/seonghapark/venv/bin/python}"
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
