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
  NPROC_PER_NODE=6 ./run_train.sh multi ./hosts.txt -- python train.py --epochs 1

Environment overrides:
  XPU_CMD            (default: xpu)
  SCHEDULER          (default: auto)
  NPROC_PER_NODE     (default: 4)
  NNODES             (multi only, default: derived from hostfile unique lines)
  NPROC              (default: NNODES * NPROC_PER_NODE for multi, else NPROC_PER_NODE)
  AUTO_RETRY         (default: 1, used for multi)
  SPARE_NODES        (default: auto, used for multi + AUTO_RETRY=1)
  FAILOVER_PROFILE   (default: auto, used for multi + AUTO_RETRY=1)
  HOST_IP_MAP        (optional JSON file path for deterministic IP->host mapping)
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
    echo "error: multi mode requires <hostfile>" >&2
    usage
    exit 1
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
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Resolve launcher relative to this repo, not the caller's cwd
LAUNCHER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# PYTHON_BIN may target a compute-node-only frameworks build; fall back for the
# launcher process itself (the training command still uses PYTHON_BIN as given).
LAUNCHER_PY="$PYTHON_BIN"
if ! command -v "$LAUNCHER_PY" >/dev/null 2>&1; then
  LAUNCHER_PY="$(command -v python3 || command -v python || true)"
fi

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
  NNODES="${NNODES:-$(awk 'NF {print $1}' "$HOSTFILE" | sort -u | wc -l)}"
  NPROC="${NPROC:-$((NNODES * NPROC_PER_NODE))}"

  LAUNCH_CMD+=("--hostfile" "$HOSTFILE" "-nh" "$NNODES" "-n" "$NPROC" "-ppn" "$NPROC_PER_NODE")

  AUTO_RETRY="${AUTO_RETRY:-1}"
  if [[ "$AUTO_RETRY" == "1" ]]; then
    SPARE_NODES="${SPARE_NODES:-auto}"
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
