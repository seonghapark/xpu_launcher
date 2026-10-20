#!/usr/bin/env bash
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -k doe
#PBS -j oe
#PBS -l filesystems=home:flare
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -N agpt_osexit
#
# ISOLATED os._exit() EXPERIMENT
#
# HYPOTHESIS
#   The teardown failure (job 8929331: 48/48 ranks SIGSEGV in a re-fault loop
#   inside free() <- ~CompressedRTDeviceBinaryImage <- descriptor_unreg <-
#   _dl_fini <- exit()) happens strictly AFTER all work. If the process leaves
#   via os._exit() instead of exit(), glibc never runs _dl_fini, the faulting
#   code never executes, and ranks terminate immediately with status 0 --
#   without touching the checkpoint, which is already durable by then.
#
# PRECONDITION, ALREADY MET
#   Job 8929331 phase 2: checkpoint step-3, 48/48 .distcp shards, RELOAD OK.
#   That is what makes skipping native teardown evaluable at all.
#
# NO SHARED CODE IS CHANGED
#   The exit is injected through a PYTHONPATH sitecustomize (osexit_exp/), the
#   same mechanism the capture runs used. trainer.py, train.py, checkpoint.py
#   and the launcher are untouched; the training invocation below is
#   byte-identical to job_stall_gdb_30m.sh phase 1. Reverting the experiment
#   means not setting AGPT_OSEXIT_EXPERIMENT=1.
#
# THE EXIT IS GUARDED (see osexit_exp/sitecustomize.py)
#   It fires only as the LAST atexit handler (registered first => LIFO last),
#   and only if: the experiment is enabled, no unhandled exception occurred,
#   the intended exit status is 0, and the process group is already destroyed
#   -- the trainer's own signal that every collective, checkpoint barrier
#   included, has completed. Otherwise it logs why and falls through to the
#   normal exit path, changing nothing.
#
# WHAT WOULD FALSIFY THE HYPOTHESIS
#   - ranks still stall  => the fault is not only in _dl_fini; os._exit() is
#                           not the answer and the Intel report stands alone.
#   - RELOAD not OK      => teardown mattered after all. That is the one
#                           outcome that would make this unsafe to adopt, and
#                           phase 2 exists to catch it.
#   - "os._exit SKIPPED" => a precondition did not hold. Read the reason; do
#                           NOT loosen the guard to force a result.

set -o pipefail

BASE=/lus/flare/projects/datascience/seonghapark
LAUNCH=${BASE}/xpu_launcher
OUT=${BASE}/converted_models/agpt-2b-v2-step-46400_hf
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=${LAUNCH}/osexit_${STAMP}
mkdir -p "$LOGDIR"

echo "=== job ${PBS_JOBID:-interactive} on $(hostname) at $(date) ==="
echo "logs: $LOGDIR"

module load frameworks/2026.1.0 || module load frameworks || {
  echo "FATAL: could not load the frameworks module" >&2; exit 1; }
export LD_LIBRARY_PATH=/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib:${LD_LIBRARY_PATH:-}
cd "$BASE" || exit 1
source venv/bin/activate

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export EVAL_BASE="${EVAL_BASE:-${BASE}/evaluation}"
export DATA_PATH="${DATA_PATH:-${EVAL_BASE}/datasets/}"
export HF_HOME="${EVAL_BASE}/models/cache"
export HF_DATASETS_CACHE="${DATA_PATH}"
export HF_HUB_DISABLE_XET=1
export TORCHINDUCTOR_CACHE_DIR=${EVAL_BASE}/.cache/inductor_${PBS_JOBID:-local}
export CCL_PROCESS_LAUNCHER=hydra
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export no_proxy="localhost,127.0.0.1"
export PYTHONMULTIPROCESSINGMETHOD=spawn
export WANDB_MODE=offline

ulimit -c unlimited
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export CCL_LOG_LEVEL=info

PHASE1_LOG=$LOGDIR/phase1_osexit.log

NODES_ALLOC=$(awk 'NF {print $1}' "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort -u | wc -l)
[[ "$NODES_ALLOC" -lt 1 ]] && NODES_ALLOC=1
PPN=12
WANT_RANKS=$(( NODES_ALLOC * PPN ))
echo "allocation: $NODES_ALLOC nodes x $PPN = $WANT_RANKS ranks"

RUN_MARKER="$LOGDIR/.run_start_marker"
touch "$RUN_MARKER"

HOSTFILE=$LOGDIR/hosts.txt
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
  awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$HOSTFILE"
else
  hostname > "$HOSTFILE"
fi
NHOSTS=$(wc -l < "$HOSTFILE")

# ---------------------------------------------------------------------------
# THE EXPERIMENT. The shim is a checked-in file, not generated here, so the
# exact code that ran is recoverable from the run directory.
# ---------------------------------------------------------------------------
SHIM="$LAUNCH/osexit_exp"
if [[ ! -f "$SHIM/sitecustomize.py" ]]; then
  echo "FATAL: missing $SHIM/sitecustomize.py" >&2; exit 1
fi
cp "$SHIM/sitecustomize.py" "$LOGDIR/sitecustomize.py.used"
echo "shim md5: $(md5sum "$SHIM/sitecustomize.py" | awk '{print $1}')"

export PYTHONPATH="${SHIM}${PYTHONPATH:+:$PYTHONPATH}"
export AGPT_OSEXIT_EXPERIMENT=1
echo "AGPT_OSEXIT_EXPERIMENT=1  (unset it and this run is the baseline)"

# Snapshot the core directory. A core appearing here would mean the fault was
# reached anyway -- the opposite of what the experiment predicts.
CORE_DIR="${LAUNCH}/xpu_torchtitan/torchtitan_repo"
CORES_BEFORE=$(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l)
echo "cores present before the run: $CORES_BEFORE"

# ---------------------------------------------------------------------------
# PHASE 1 -- same training invocation as the capture job.
# ---------------------------------------------------------------------------
echo
echo "########## PHASE 1: train, then exit via os._exit() ##########"
# 600s is the same budget the capture job used. A healthy run reaches teardown
# at ~90s; the stall under study runs 350-470s. So if this job is still inside
# phase 1 much past ~120s, the workaround did NOT work -- and the timeout
# still bounds it.
PHASE1_TIMEOUT=${PHASE1_TIMEOUT:-600}
echo "phase 1 timeout: ${PHASE1_TIMEOUT}s"
PHASE1_START=$(date +%s)
(
  cd "${LAUNCH}/xpu_torchtitan" || exit 1
  MODEL="$OUT" \
  MODULE=agpt \
  CONFIG=agpt_2b_yarn \
  DATASET_NAME=PG19 \
  DATASET_PATH=${BASE}/assets/hf/datasets/pg19/data \
  CKPT_FOLDER="osexit" \
  timeout --signal=TERM --kill-after=120 "$PHASE1_TIMEOUT" ./run_train_agpt.sh --model 2b multi \
    -- \
    --training.steps 3 \
    --training.seq_len 16384 \
    --training.local_batch_size 1 \
    --checkpoint.interval 1000
) 2>&1 | tee "$PHASE1_LOG"
RC1=${PIPESTATUS[0]}
PHASE1_END=$(date +%s)
PHASE1_ELAPSED=$(( PHASE1_END - PHASE1_START ))
echo "phase 1 exit: $RC1  (wall: ${PHASE1_ELAPSED}s)"

# Teardown cost: last application log line -> process exit. This is the number
# the experiment is trying to drive to ~0 (it was ~505s in job 8929331).
LAST_LOG_EPOCH=$(grep -oE "^\[titan\] [0-9-]+ [0-9:]+" "$PHASE1_LOG" 2>/dev/null \
  | tail -1 | sed 's/^\[titan\] //' | { read -r d t && date -d "$d $t" +%s 2>/dev/null || true; })
STALL=""
if [[ -n "$LAST_LOG_EPOCH" ]]; then
  STALL=$(( PHASE1_END - LAST_LOG_EPOCH ))
  echo "teardown: ~${STALL}s between last log line and process exit"
fi

# ---------------------------------------------------------------------------
# Did every rank actually leave? os._exit() cannot be partial, but a rank that
# never reached the handler could still be alive. Ask each node directly
# rather than trusting the launcher's summary, which under-reports.
# ---------------------------------------------------------------------------
echo
echo "########## survivors: any training process still alive? ##########"
# Scope this precisely, or it reports survivors that do not exist. In job
# 8929490 the pattern 'train.py|torchtitan' matched 5 unrelated processes --
# another user's grep over a torchtitan path, and leftover decoys from watcher
# testing -- and the report claimed 5 survivors when the ranks were all gone.
# Two fixes: restrict to OUR uid, and match the actual rank argv
# (run_train_agpt / titan_train.py) rather than any string containing
# "torchtitan". -x on the ssh side is not enough; pgrep -f also sees the ssh
# command line itself on some paths, so exclude pgrep/bash/ssh explicitly.
MY_UID=$(id -u)
SURVIVORS=0
SURV_UNKNOWN=0
while read -r h; do
  [[ -z "$h" ]] && continue
  short=${h%%.*}
  out=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 "$h" \
        "pgrep -u $MY_UID -af 'titan_train[.]py|run_train_agpt' 2>/dev/null \
           | grep -vE 'pgrep|ssh |bash -c' | wc -l" </dev/null 2>/dev/null | tr -dc '0-9')
  if [[ -z "$out" ]]; then
    # An unreachable node is NOT zero survivors. Say so rather than silently
    # crediting it as clean.
    echo "  $short: UNREACHABLE (cannot confirm; nodes may already be released)"
    SURV_UNKNOWN=$(( SURV_UNKNOWN + 1 ))
  else
    echo "  $short: $out surviving process(es)"
    SURVIVORS=$(( SURVIVORS + out ))
  fi
done < "$HOSTFILE"
echo "total survivors: $SURVIVORS (expect 0)${SURV_UNKNOWN:+   unreachable nodes: $SURV_UNKNOWN}"

CORES_AFTER=$(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l)
echo "cores after the run: $CORES_AFTER (was $CORES_BEFORE)"

# ---------------------------------------------------------------------------
# PHASE 2 -- the checkpoint must be no worse than without the experiment.
# Identical probe to job_stall_gdb_30m.sh, so the two are comparable.
# ---------------------------------------------------------------------------
echo
echo "########## PHASE 2: is the checkpoint durable and reloadable? ##########"
CKPT_ROOT="${LAUNCH}/xpu_torchtitan/outputs"
LATEST=""
while IFS= read -r d; do
  [[ -d "$d" ]] || continue
  if [[ -z "$RUN_MARKER" || "$d" -nt "$RUN_MARKER" ]]; then LATEST="$d"; break; fi
done < <(find "$CKPT_ROOT" -mindepth 3 -maxdepth 3 -type d -name 'step-*' \
           -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
echo "checkpoint under test: ${LATEST:-NONE (none written by this run)}"

if [[ -z "$LATEST" ]]; then
  echo "SKIP: no checkpoint produced by this run."
  RC2=2
else
  NSHARD=$(ls "$LATEST"/*.distcp 2>/dev/null | wc -l)
  echo "shards: $NSHARD (expect $WANT_RANKS)   metadata: $([[ -f "$LATEST/.metadata" ]] && echo present || echo MISSING)"
  python3 - "$LATEST" "$WANT_RANKS" <<'RELOAD'
import os, sys
path, want = sys.argv[1], int(sys.argv[2])
shards = [f for f in os.listdir(path) if f.endswith(".distcp")]
missing = [i for i in range(want) if f"__{i}_0.distcp" not in shards]
print(f"  shard count   : {len(shards)}/{want}")
print(f"  missing ranks : {missing if missing else 'none'}")
meta = os.path.join(path, ".metadata")
print(f"  .metadata     : {'present %d bytes' % os.path.getsize(meta) if os.path.exists(meta) else 'MISSING'}")
try:
    from torch.distributed.checkpoint import FileSystemReader
    md = FileSystemReader(path).read_metadata()
    n = len(md.state_dict_metadata)
    print(f"  readable keys : {n}")
    print("  RELOAD OK" if n and not missing else "  RELOAD INCOMPLETE")
    sys.exit(0 if (n and not missing) else 1)
except Exception as exc:
    print(f"  RELOAD FAILED: {exc!r}")
    sys.exit(1)
RELOAD
  RC2=$?
fi

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
echo
echo "================= os._exit() EXPERIMENT REPORT ================="

RANKS_DONE=$(grep -oE "Training completed" "$PHASE1_LOG" 2>/dev/null | wc -l)
RANKS_PG=$(grep -oE "Process group destroyed" "$PHASE1_LOG" 2>/dev/null | wc -l)
RANKS_ATEXIT=$(grep -oE "SHUTDOWN rank=[0-9]+ atexit LAST" "$PHASE1_LOG" 2>/dev/null \
  | awk '{print $2}' | sort -u | wc -l)
RANKS_EXIT=$(grep -oE "SHUTDOWN rank=[0-9]+ os\._exit\([0-9]+\) NOW" "$PHASE1_LOG" 2>/dev/null \
  | awk '{print $2}' | sort -u | wc -l)
RANKS_SKIP=$(grep -oE "SHUTDOWN rank=[0-9]+ os\._exit SKIPPED" "$PHASE1_LOG" 2>/dev/null \
  | awk '{print $2}' | sort -u | wc -l)

echo "-- work (OBSERVED) --"
echo "  'Training completed'                    : $RANKS_DONE / $WANT_RANKS"
echo "  'Process group destroyed'               : $RANKS_PG / $WANT_RANKS"
echo "  distinct ranks reaching final atexit    : $RANKS_ATEXIT / $WANT_RANKS"
echo
echo "-- the experiment --"
echo "  ranks that took os._exit()              : $RANKS_EXIT / $WANT_RANKS"
echo "  ranks that SKIPPED it                   : $RANKS_SKIP"
if (( RANKS_SKIP > 0 )); then
  echo "  reasons given:"
  grep -oE "os\._exit SKIPPED \(([^)]*\)?[^)]*)" "$PHASE1_LOG" 2>/dev/null \
    | sed 's/^os\._exit SKIPPED //' | sort | uniq -c | sed 's/^/    /'
fi
echo "  surviving processes after exit          : $SURVIVORS (expect 0)"
(( SURV_UNKNOWN > 0 )) && echo "  nodes that could not be checked         : $SURV_UNKNOWN (survivor count is a LOWER BOUND)"
echo "  new core dumps                          : $(( CORES_AFTER - CORES_BEFORE )) (expect 0)"
[[ -n "$STALL" ]] && echo "  teardown time                           : ~${STALL}s  (was ~505s in job 8929331)"
echo "  launcher exit status                    : $RC1  (0 = clean; 124 = our timeout = still stalling)"
case "${RC2:-}" in
  0) echo "  checkpoint durable + reloadable         : YES (RELOAD OK)" ;;
  2) echo "  checkpoint durable + reloadable         : N/A (none written)" ;;
  *) echo "  checkpoint durable + reloadable         : NO  (see phase 2)" ;;
esac

echo
echo "-- verdict --"
if (( RANKS_EXIT == WANT_RANKS )) && (( SURVIVORS == 0 )) && (( SURV_UNKNOWN == 0 )) \
   && [[ "${RC2:-1}" == "0" ]] && (( RC1 == 0 )); then
  echo "  WORKS: all $WANT_RANKS ranks exited via os._exit(), none survived,"
  echo "  no new cores, checkpoint reloads. The teardown fault was skipped."
  echo "  This is evidence for a workaround -- NOT a fix. The Intel defect is"
  echo "  unchanged and the bug report still stands."
elif (( RC1 == 124 )); then
  echo "  DID NOT WORK: exit 124 is our own timeout, so ranks were still"
  echo "  stalling. The fault is not confined to the _dl_fini path."
elif grep -q "@jit functions should be defined in a Python file" "$PHASE1_LOG" 2>/dev/null; then
  echo "  NOT TESTED: the run hit the intermittent Inductor/Triton first-compile"
  echo "  failure ('@jit functions should be defined in a Python file') and died"
  echo "  before teardown, so the os._exit() path never executed. This is the"
  echo "  known flake from job 8929205, NOT a result about the experiment."
  echo "  The guard behaved correctly by refusing to fire on a failing run."
  echo "  Resubmit; do not read anything into the counters above."
else
  echo "  INCONCLUSIVE: read the counters above before drawing a conclusion."
  echo "  In particular a nonzero SKIPPED count means a precondition did not"
  echo "  hold; fix the precondition, do not loosen the guard."
fi

echo
echo "artifacts:"
echo "  training log : $PHASE1_LOG"
echo "  shim as run  : $LOGDIR/sitecustomize.py.used"
echo "=== done at $(date) ==="
