#!/usr/bin/env bash
# Convert a DCP checkpoint to HF safetensors, then launch training from it.
#
# This wraps the env-var invocation style of xpu_torchtitan/run_train_agpt.sh.
# MODEL may point at either format:
#   - already HF (config.json + *.safetensors) -> used as-is, no conversion
#   - DCP (a step-NNNN dir with .metadata)     -> converted first, then trained
#
# Conversion goes through dcp_to_hf.py, which reuses TorchTitan's own
# state_dict_adapter rather than renaming keys by hand. That is what makes the
# Q/K rotary layout come out right for the model actually being trained; see
# the header of dcp_to_hf.py.
#
# Usage mirrors run_train_agpt.sh exactly -- same env vars, same positional
# mode, same "--" separator for torchtitan args:
#
#   MODEL=/path/to/checkpoint/step-1000 \
#   MODULE=agpt CONFIG=agpt_2b_yarn \
#   DATASET_NAME=PG19 DATASET_PATH=/path/to/pg19/data \
#   ./run_convert_and_train.sh multi \
#     --resource-monitor \
#     -- \
#     --training.steps 400 --training.seq_len 16384
#
# Extra env vars beyond run_train_agpt.sh's:
#   CONVERT_OUTPUT_DIR  where converted models land (default ./converted_models)
#   CONVERT_ASSETS_FROM dir to copy config.json/tokenizer from. Required when
#                       converting, since a DCP checkpoint has neither.
#   FORCE_CONVERT=1     reconvert even if the output already exists
#   SKIP_CONVERT=1      never convert; fail if MODEL is not already HF
#   DRY_RUN=1           print what would run, do nothing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/dcp_to_hf.py"
TRAINER="${SCRIPT_DIR}/xpu_torchtitan/run_train_agpt.sh"

CONVERT_OUTPUT_DIR="${CONVERT_OUTPUT_DIR:-${SCRIPT_DIR}/converted_models}"
FORCE_CONVERT="${FORCE_CONVERT:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
DRY_RUN="${DRY_RUN:-0}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[convert-and-train] $*" >&2; }
show_cmd() { printf '[DRY RUN]'; printf ' %q' "$@"; echo; }

[[ -f "$CONVERTER" ]] || die "converter not found: $CONVERTER"
[[ -x "$TRAINER"   ]] || die "trainer not found or not executable: $TRAINER"

MODEL_PATH="${MODEL:-${MODEL_PATH:-}}"
[[ -n "$MODEL_PATH" ]] || die "MODEL (or MODEL_PATH) must be set to the checkpoint directory"
[[ -d "$MODEL_PATH" ]] || die "model path not found: $MODEL_PATH"
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"

# Same interpreter resolution as run_train.sh: an activated venv first, then the
# site venv that actually has torch. A bare python3 on an Aurora login node is
# /usr/bin/python3 (3.6, no torch), so it is the last resort.
XPU_VENV="${XPU_VENV:-/lus/flare/projects/datascience/seonghapark/venv}"
resolve_python_bin() {
  local candidate
  for candidate in \
    "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" \
    "${SCRIPT_DIR}/.venv/bin/python" \
    "${XPU_VENV}/bin/python"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done
  command -v python3 2>/dev/null || echo python3
}
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

is_hf_format() {
  [[ -f "$1/config.json" ]] || return 1
  compgen -G "$1/*.safetensors" >/dev/null 2>&1 && return 0
  [[ -f "$1/pytorch_model.bin" ]]
}

is_dcp_format() {
  # TorchTitan DCP checkpoints always carry a .metadata file written by
  # FileSystemWriter; the *.distcp shards alone are not a reliable marker.
  [[ -f "$1/.metadata" ]]
}

# Split our args from the torchtitan args after "--". Everything before the
# separator is forwarded to run_train_agpt.sh unchanged (mode, --resource-*,
# etc.); everything after is forwarded after its own "--", and is ALSO what the
# converter needs so it builds the same model config training will build.
PRE_ARGS=()
TITAN_ARGS=()
seen_sep=0
for arg in "$@"; do
  if [[ $seen_sep -eq 0 && "$arg" == "--" ]]; then
    seen_sep=1
    continue
  fi
  if [[ $seen_sep -eq 0 ]]; then
    PRE_ARGS+=("$arg")
  else
    TITAN_ARGS+=("$arg")
  fi
done

# --------------------------------------------------------------------------
# Step 1: convert if needed
# --------------------------------------------------------------------------
FINAL_MODEL_PATH="$MODEL_PATH"

if is_hf_format "$MODEL_PATH"; then
  log "MODEL is already HF format, skipping conversion: $MODEL_PATH"
elif ! is_dcp_format "$MODEL_PATH"; then
  die "$MODEL_PATH is neither HF (config.json + *.safetensors) nor DCP (.metadata)"
elif [[ "$SKIP_CONVERT" == "1" ]]; then
  die "MODEL is DCP format but SKIP_CONVERT=1"
else
  CONVERTED="${CONVERT_OUTPUT_DIR}/$(basename "$MODEL_PATH")_hf"

  # The converter needs --module/--config to pick the right adapter. They come
  # from the env (as with run_train_agpt.sh) or from the torchtitan args.
  CONV_TITAN_ARGS=("${TITAN_ARGS[@]}")
  has_titan_arg() {
    local needle="$1" a
    for a in "${TITAN_ARGS[@]+"${TITAN_ARGS[@]}"}"; do
      [[ "$a" == "$needle" || "$a" == "${needle}="* ]] && return 0
    done
    return 1
  }
  if ! has_titan_arg "--module"; then
    [[ -n "${MODULE:-}" ]] || die "MODULE must be set (or pass --module in the torchtitan args)"
    CONV_TITAN_ARGS+=("--module" "$MODULE")
  fi
  if ! has_titan_arg "--config"; then
    [[ -n "${CONFIG:-}" ]] || die "CONFIG must be set (or pass --config in the torchtitan args)"
    CONV_TITAN_ARGS+=("--config" "$CONFIG")
  fi

  # A DCP checkpoint has no tokenizer or config.json, but training needs both
  # from --hf_assets_path, so the converter has to copy them from somewhere.
  ASSETS_FROM="${CONVERT_ASSETS_FROM:-}"
  [[ -n "$ASSETS_FROM" ]] || die \
    "CONVERT_ASSETS_FROM must point at an HF directory to copy config.json and
       tokenizer files from (a DCP checkpoint contains neither, and training
       reads both from --hf_assets_path). Use the base model directory."
  [[ -d "$ASSETS_FROM" ]] || die "CONVERT_ASSETS_FROM not found: $ASSETS_FROM"
  ASSETS_FROM="$(cd "$ASSETS_FROM" && pwd)"

  CONVERT_CMD=(
    "$PYTHON_BIN" "$CONVERTER"
    --dcp-path "$MODEL_PATH"
    --output "$CONVERTED"
    --assets-from "$ASSETS_FROM"
  )
  [[ "$FORCE_CONVERT" == "1" ]] && CONVERT_CMD+=(--force)
  CONVERT_CMD+=(-- "${CONV_TITAN_ARGS[@]}")

  if is_hf_format "$CONVERTED" && [[ "$FORCE_CONVERT" != "1" ]]; then
    log "reusing existing conversion: $CONVERTED (FORCE_CONVERT=1 to redo)"
  else
    log "converting DCP -> HF"
    log "  from: $MODEL_PATH"
    log "  to:   $CONVERTED"
    if [[ "$DRY_RUN" == "1" ]]; then
      show_cmd "${CONVERT_CMD[@]}"
    else
      "${CONVERT_CMD[@]}" || die "conversion failed"
      is_hf_format "$CONVERTED" || die "conversion reported success but $CONVERTED is not valid HF format"
    fi
  fi

  FINAL_MODEL_PATH="$CONVERTED"
fi

# --------------------------------------------------------------------------
# Step 2: train
# --------------------------------------------------------------------------
# run_train_agpt.sh needs the model SIZE (2b|20b|80b) as well as the path, and
# when --model is absent it infers the size from the path basename via
# "*-2b-*"-style globs. A DCP checkpoint is typically named step-NNNN, which
# carries no size, so that inference fails with "invalid model: ". Derive the
# size ourselves and pass --model explicitly. CONFIG is the reliable source
# (agpt_2b_yarn -> 2b); fall back to the ORIGINAL model path (still unclobbered
# here) for the odd config name that omits the size.
if ! printf '%s\n' "${PRE_ARGS[@]+"${PRE_ARGS[@]}"}" | grep -qx -- "--model"; then
  MODEL_SIZE=""
  case "${CONFIG:-}" in
    *_2b_*|*_2b)   MODEL_SIZE="2b" ;;
    *_20b_*|*_20b) MODEL_SIZE="20b" ;;
    *_80b_*|*_80b) MODEL_SIZE="80b" ;;
  esac
  if [[ -z "$MODEL_SIZE" ]]; then
    case "$(basename "$MODEL_PATH")" in
      *-2b-*|*-2b|2b*)    MODEL_SIZE="2b" ;;
      *-20b-*|*-20b|20b*) MODEL_SIZE="20b" ;;
      *-80b-*|*-80b|80b*) MODEL_SIZE="80b" ;;
    esac
  fi
  if [[ -n "$MODEL_SIZE" ]]; then
    log "model size: $MODEL_SIZE"
    PRE_ARGS+=(--model "$MODEL_SIZE")
  else
    log "WARNING: could not infer model size from CONFIG='${CONFIG:-}' or" \
        "$(basename "$MODEL_PATH"); pass --model 2b|20b|80b if training rejects it"
  fi
fi

# MODEL is how run_train_agpt.sh receives the model path. Exported after the
# size inference above, which reads the pre-conversion MODEL_PATH.
export MODEL="$FINAL_MODEL_PATH"
export MODEL_PATH="$FINAL_MODEL_PATH"

TRAIN_CMD=("$TRAINER" "${PRE_ARGS[@]+"${PRE_ARGS[@]}"}")
if [[ ${#TITAN_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=(-- "${TITAN_ARGS[@]}")
fi

log "training with MODEL=$FINAL_MODEL_PATH"
if [[ "$DRY_RUN" == "1" ]]; then
  show_cmd "${TRAIN_CMD[@]}"
  exit 0
fi

exec "${TRAIN_CMD[@]}"
