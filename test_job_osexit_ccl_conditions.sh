#!/bin/bash
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -N osexit_ccl
#
# ===========================================================================
# IS os._exit() STILL NEEDED?  CCL_OP_SYNC / PROCESS_LAUNCHER AS CONFOUNDERS
# ===========================================================================
#
# THE QUESTION
#   The teardown SIGSEGV (sycl.descriptor_unreg in _dl_fini) first appeared
#   2026-10-13. Commit 153af9c "remove ezpz" (2026-10-08) deleted, tree-wide,
#   49x `export CCL_OP_SYNC=1` and 10x `export CCL_PROCESS_LAUNCHER=pmix`.
#   Nothing re-added them; oneCCL's own env dump from job 8929490 confirms the
#   current effective state is `CCL_OP_SYNC: 0` / `CCL_PROCESS_LAUNCHER: hydra`.
#   So both settings flipped five days BEFORE the fault was first seen. This
#   job tests whether restoring them removes the need for the os._exit() shim.
#
# WHY EACH CONDITION RUNS WITH THE SHIM **OFF**
#   The shim's entire function is to os._exit() before glibc reaches the
#   faulting path. With it ON, every arm reports a clean shutdown whether or
#   not CCL_OP_SYNC fixed anything -- the measurement would be vacuous. So the
#   measurement arms run shim OFF, and a smaller number of shim-ON trials run
#   alongside as the safety control (confirming the shim still works, and that
#   nothing else about today's machine state changed).
#
# CONDITIONS
#   A  hydra, CCL_OP_SYNC unset   shim OFF   positive control: must REPRODUCE
#                                            the fault, or the job proves
#                                            nothing about the other arms.
#   B  hydra, CCL_OP_SYNC=1       shim OFF   test 1 (the user's test)
#   C  pmix,  CCL_OP_SYNC=1       shim OFF   test 2 -- the pre-10-08 config,
#                                            gated on the P probe below.
#   S  hydra, CCL_OP_SYNC unset   shim ON    safety control
#
# HOW MANY TRIALS ARE ENOUGH HERE
#   Fewer than the Inductor campaign needed, and for a principled reason. That
#   flake was PROBABILISTIC (~40%), so one clean run meant nothing and 35
#   trials were required. This fault is DETERMINISTIC: gdb caught 48/48 ranks
#   on the same stack, and every shim-OFF 48-rank run to date has hit it. For
#   a deterministic failure, presence/absence is visible in 2-3 trials; the
#   repeats guard against a flaky node, not against sampling noise. That is
#   fortunate, because at ~505s of teardown per shim-OFF trial the hour holds
#   only about 5-6 trials total. Any arm that ends with n<2 is called
#   UNDERPOWERED in the report rather than being read as a result.
#
#   P  pmix acceptance probe, run FIRST, shim ON, one trial. `pmix` is NOT a
#      standalone token in the launcher enum of any oneCCL on this machine
#      (hydra/mpich/none only, across 2021.17 and 2022.1), and libccl.so has an
#      "unexpected process launcher" rejection path. If oneCCL rejects it, that
#      is a CONFIGURATION/COMPATIBILITY failure -- condition C is dropped and
#      recorded as such, never as evidence about the runtime defect. The probe
#      runs first so a rejection costs ~3 minutes, not the hour.
#
#   torchrun is NOT tested as a CCL_PROCESS_LAUNCHER value: the string does not
#   occur in libccl.so at all, and the xpu CLI has no torchrun dispatch
#   (src/cli/launch.py:523 handles mpiexec/mpirun/srun; :212 mentions torchrun
#   only in a help string). Reported as not-implemented with that evidence
#   rather than burned as a trial.
#
# CORE DUMP GUARD -- READ BEFORE CHANGING
#   Shim OFF means the SIGSEGV is no longer skipped. Job 8924552 wrote 47 cores
#   averaging 11.3 GB (529.8 GB total), so a 48-rank trial would write ~542 GB.
#   That is not only a quota problem: the write lands INSIDE the teardown
#   window and would inflate shutdown_s, the very metric this job measures.
#
#   The cap is applied IN-PROCESS by the shim (AGPT_CORE_LIMIT_BYTES), NOT by
#   `ulimit -c` here. PALS does not propagate rlimits to ranks by default --
#   that is what its --rlimits flag exists for -- so a shell-level limit would
#   stay in this shell while every rank dumped at `unlimited`. Verified: with
#   the cap set, a real SIGSEGV produces a 2.3 MB core instead of ~11 GB.
#   The metric asked for is "new core dumps", i.e. how many, not how big.
#
# HELD CONSTANT ACROSS ALL CONDITIONS (no confounders)
#   per-rank inductor cache isolation ON, fresh+verified-empty root per trial,
#   same model/steps/seq_len, same 4 nodes, same 12 ppn. Only the named
#   variable moves.
# ===========================================================================

set -o pipefail

BASE=/lus/flare/projects/datascience/seonghapark
LAUNCH=${BASE}/xpu_launcher
OUT=${BASE}/converted_models/agpt-2b-v2-step-46400_hf
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=${LAUNCH}/osexit_ccl_${STAMP}
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
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export no_proxy="localhost,127.0.0.1"
export PYTHONMULTIPROCESSINGMETHOD=spawn
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# See CORE DUMP GUARD above. The hard limit must stay high enough for the
# shim's in-process setrlimit to land under it; the shim lowers the soft limit
# on every rank, which is what actually bounds the dump.
ulimit -c unlimited
export AGPT_CORE_LIMIT_BYTES=2097152   # 2 MB per rank

# CCL_PROCESS_LAUNCHER and CCL_OP_SYNC are deliberately NOT exported here.
# They are the independent variables and are set per-condition in run_trial.

# Dependencies are already present in the venv (verified before submit). Do not
# pip install from inside a 4-node allocation: it burns node-hours on a login
# task and can rewrite the environment mid-campaign.
python3 - <<'DEPS' || { echo "FATAL: dependency check failed" >&2; exit 1; }
import importlib.util as u, sys
missing = [m for m in ("tyro", "typeguard", "tensorboard", "torch") if u.find_spec(m) is None]
print("deps missing:", missing if missing else "none")
sys.exit(1 if missing else 0)
DEPS

SHIM="$LAUNCH/inductor_iso"
[[ -f "$SHIM/sitecustomize.py" ]] || { echo "FATAL: missing $SHIM/sitecustomize.py" >&2; exit 1; }
[[ -f "$LAUNCH/osexit_exp/sitecustomize.py" ]] || { echo "FATAL: missing osexit_exp shim" >&2; exit 1; }
cp "$SHIM/sitecustomize.py" "$LOGDIR/sitecustomize.py.used"
cp "$LAUNCH/osexit_exp/sitecustomize.py" "$LOGDIR/osexit_sitecustomize.py.used"
echo "iso shim md5   : $(md5sum "$SHIM/sitecustomize.py" | awk '{print $1}')"
echo "osexit shim md5: $(md5sum "$LAUNCH/osexit_exp/sitecustomize.py" | awk '{print $1}')"
export PYTHONPATH="${SHIM}${PYTHONPATH:+:$PYTHONPATH}"

# Inductor isolation is held ON for every trial (proven mitigation, job
# 8931618). It is NOT an independent variable here.
export AGPT_INDUCTOR_ISOLATE=1

NODES_ALLOC=$(awk 'NF {print $1}' "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | sort -u | wc -l)
[[ "$NODES_ALLOC" -lt 1 ]] && NODES_ALLOC=1
PPN=12
WANT_RANKS=$(( NODES_ALLOC * PPN ))
HOSTFILE=$LOGDIR/hosts.txt
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
  awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$HOSTFILE"
else
  hostname > "$HOSTFILE"
fi
MY_UID=$(id -u)
echo "allocation: $NODES_ALLOC nodes x $PPN = $WANT_RANKS ranks"

CORE_DIR="${LAUNCH}/xpu_torchtitan/torchtitan_repo"
CACHE_ROOT="${EVAL_BASE}/.cache/osexit_ccl_${PBS_JOBID:-local}"
mkdir -p "$CACHE_ROOT"

# TRIAL BUDGET -- the 300s I first wrote here was WRONG and would have
# destroyed the measurement. A healthy run reaches "Training completed" at
# ~70s, but the teardown fault runs 350-470s and the measured baseline
# teardown was ~505s (job 8929540). A 300s cap would kill every shim-OFF
# trial mid-fault, turning shutdown_s -- one of the six requested metrics --
# into a censored constant and misfiling every TEARDOWN_SEGV as a
# TEARDOWN_HANG. 600s clears the ~505s baseline with margin.
TRIAL_TIMEOUT=${TRIAL_TIMEOUT:-600}
# Never START a trial that could run past the reporting reserve. The previous
# campaign was killed mid-report at walltime; this is the guard against that.
# Budget the WORST case of a trial that has just started:
#   TRIAL_TIMEOUT (600) + kill-after (120) + post-trial checks (~120: survivor
#   ssh over 4 hosts at 20s each, plus checkpoint validation) + report (180).
REPORT_RESERVE_S=${REPORT_RESERVE_S:-180}
RESERVE_S=$(( TRIAL_TIMEOUT + 120 + 120 + REPORT_RESERVE_S ))
JOB_START=$(date +%s)
WALL_BUDGET=${WALL_BUDGET:-3600}

RESULTS="$LOGDIR/results.tsv"
printf 'trial\tcond\tlauncher\topsync\tshim\tverdict\trc\twall_s\tcompile_s\tshutdown_s\tcompleted\texit_ranks\tnew_cores\tsurvivors\tckpt\n' > "$RESULTS"

# ---------------------------------------------------------------------------
# run_trial <n> <cond> <launcher> <opsync:0|1> <shim:on|off>
# ---------------------------------------------------------------------------
run_trial() {
  local n="$1" cond="$2" launcher="$3" opsync="$4" shim="$5"
  local log="$LOGDIR/trial${n}_${cond}.log"
  local tcache="$CACHE_ROOT/t${n}_${cond}"

  echo
  echo "########## TRIAL $n  [cond $cond]  launcher=$launcher op_sync=$opsync shim=$shim ##########"

  # Fresh cache, verified empty. The @jit race is a FIRST-compile race; a warm
  # cache means no compile and a vacuous pass. Refuse rather than mis-report.
  if [[ -e "$tcache" ]] && [[ -n "$(ls -A "$tcache" 2>/dev/null)" ]]; then
    echo "  ABORT: cache root $tcache is not empty; trial would not compile."
    printf '%s\t%s\t%s\t%s\t%s\tABORT_WARM_CACHE\t-\t-\t-\t-\t-\t-\t-\t-\t-\n' \
      "$n" "$cond" "$launcher" "$opsync" "$shim" >> "$RESULTS"
    return
  fi
  mkdir -p "$tcache"
  export TORCHINDUCTOR_CACHE_DIR="$tcache"

  export CCL_PROCESS_LAUNCHER="$launcher"
  if [[ "$opsync" == "1" ]]; then export CCL_OP_SYNC=1; else unset CCL_OP_SYNC; fi
  if [[ "$shim" == "on" ]]; then export AGPT_OSEXIT_EXPERIMENT=1; else unset AGPT_OSEXIT_EXPERIMENT; fi
  # Make oneCCL print its resolved env so the manipulation is VERIFIED, not
  # assumed -- the same discipline that made the isolation result trustworthy.
  export CCL_LOG_LEVEL=info

  local cores_before
  cores_before=$(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l)

  local t0 t1 rc
  t0=$(date +%s)
  (
    cd "${LAUNCH}/xpu_torchtitan" || exit 1
    MODEL="$OUT" \
    MODULE=agpt \
    CONFIG=agpt_2b_yarn \
    DATASET_NAME=PG19 \
    DATASET_PATH=${BASE}/assets/hf/datasets/pg19/data \
    CKPT_FOLDER="occl_t${n}_${cond}" \
    timeout --signal=TERM --kill-after=120 "$TRIAL_TIMEOUT" ./run_train_agpt.sh --model 2b multi \
      -- \
      --training.steps 3 \
      --training.seq_len 16384 \
      --training.local_batch_size 1 \
      --checkpoint.interval 1000
  ) 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  local wall=$(( t1 - t0 ))

  # --- VERIFY the manipulation actually reached oneCCL ------------------------
  # A trial whose env did not take is not a trial of that condition.
  local eff_launcher eff_opsync
  eff_launcher=$(grep -m1 -oE 'CCL_PROCESS_LAUNCHER: [a-z]+' "$log" 2>/dev/null | awk '{print $2}')
  eff_opsync=$(grep -m1 -oE 'CCL_OP_SYNC: [0-9]+' "$log" 2>/dev/null | awk '{print $2}')
  [[ -z "$eff_launcher" ]] && eff_launcher="?"
  [[ -z "$eff_opsync" ]] && eff_opsync="?"
  echo "  oneCCL reports: launcher=$eff_launcher op_sync=$eff_opsync (asked $launcher / $opsync)"

  # --- did oneCCL REJECT the launcher value? ----------------------------------
  local rejected=0
  grep -qi 'unexpected process launcher' "$log" 2>/dev/null && rejected=1
  [[ "$eff_launcher" != "?" && "$eff_launcher" != "$launcher" ]] && rejected=1

  # --- compile cost: "Training starts" -> first "step:  1" --------------------
  local cs ss compile="-" tc t_completed shutdown="-"
  ss=$(grep -m1 'Training starts' "$log" 2>/dev/null | grep -oE '^\[titan\] [0-9-]+ [0-9:]+' | sed 's/^\[titan\] //')
  cs=$(grep -m1 -E 'step: +1 ' "$log" 2>/dev/null | grep -oE '^\[titan\] [0-9-]+ [0-9:]+' | sed 's/^\[titan\] //')
  if [[ -n "$ss" && -n "$cs" ]]; then
    local a b; a=$(date -d "$ss" +%s 2>/dev/null); b=$(date -d "$cs" +%s 2>/dev/null)
    [[ -n "$a" && -n "$b" ]] && compile=$(( b - a ))
  fi

  # --- shutdown time: last "Training completed" -> launcher exit --------------
  # This is the metric the teardown fault actually moves.
  tc=$(grep 'Training completed' "$log" 2>/dev/null | tail -1 | grep -oE '^\[titan\] [0-9-]+ [0-9:]+' | sed 's/^\[titan\] //')
  if [[ -n "$tc" ]]; then
    t_completed=$(date -d "$tc" +%s 2>/dev/null)
    [[ -n "$t_completed" ]] && shutdown=$(( t1 - t_completed ))
  fi

  local completed exits newcores
  completed=$(grep -c 'Training completed' "$log" 2>/dev/null)
  exits=$(grep -oE 'SHUTDOWN rank=[0-9]+ os\._exit\([0-9]+\) NOW' "$log" 2>/dev/null \
          | awk '{print $2}' | sort -u | wc -l)
  newcores=$(( $(ls -1 "$CORE_DIR"/core.* 2>/dev/null | wc -l) - cores_before ))

  # --- surviving processes ----------------------------------------------------
  # Scope to our uid and match the rank argv. An unreachable node is reported
  # as UNREACHABLE, never credited as clean.
  # Fanned out in PARALLEL: serial ssh at 20s x 4 hosts cost up to 80s per
  # trial, and with a ~505s teardown fault the hour only holds a handful of
  # trials -- that overhead was coming straight out of sample size.
  local survivors=0 unreachable=0 h c sdir
  sdir="$LOGDIR/.surv_t${n}"
  mkdir -p "$sdir"
  while read -r h; do
    [[ -z "$h" ]] && continue
    ( timeout 20 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$h" \
        "pgrep -u $MY_UID -af 'titan_train[.]py|run_train_agpt' 2>/dev/null | wc -l" \
        > "$sdir/$h" 2>/dev/null ) &
  done < "$HOSTFILE"
  wait
  while read -r h; do
    [[ -z "$h" ]] && continue
    c=$(cat "$sdir/$h" 2>/dev/null | tr -dc '0-9')
    if [[ -z "$c" ]]; then unreachable=$(( unreachable + 1 )); else survivors=$(( survivors + c )); fi
  done < "$HOSTFILE"
  rm -rf "$sdir"
  local surv_str="$survivors"
  (( unreachable > 0 )) && surv_str="${survivors}+${unreachable}UNREACH"

  # --- checkpoint integrity ---------------------------------------------------
  local ckpt="-" latest=""
  latest=$(find "${LAUNCH}/xpu_torchtitan/outputs" -mindepth 3 -maxdepth 3 -type d -name 'step-*' \
             -newermt "@$t0" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  if [[ -n "$latest" ]]; then
    if python3 - "$latest" "$WANT_RANKS" >>"$log" 2>&1 <<'RELOAD'
import os, sys
path, want = sys.argv[1], int(sys.argv[2])
shards = [f for f in os.listdir(path) if f.endswith(".distcp")]
missing = [i for i in range(want) if f"__{i}_0.distcp" not in shards]
print(f"  shard count   : {len(shards)}/{want}")
print(f"  missing ranks : {missing if missing else 'none'}")
try:
    from torch.distributed.checkpoint import FileSystemReader
    n = len(FileSystemReader(path).read_metadata().state_dict_metadata)
    print(f"  readable keys : {n}")
    sys.exit(0 if (n and not missing) else 1)
except Exception as exc:
    print(f"  RELOAD FAILED: {exc!r}")
    sys.exit(1)
RELOAD
    then ckpt="OK"; else ckpt="BAD"; fi
  else
    ckpt="NONE"
  fi

  # --- classify ---------------------------------------------------------------
  # Order matters. A launcher oneCCL refused, or a run killed by the Inductor
  # flake, must be named explicitly -- neither is evidence about teardown.
  local verdict
  if (( rejected )); then
    verdict="LAUNCHER_REJECTED"
  elif grep -q '@jit functions should be defined in a Python file' "$log" 2>/dev/null; then
    verdict="INDUCTOR_FLAKE"
  elif (( completed < WANT_RANKS )) && (( rc != 0 )); then
    verdict="FAILED_BEFORE_TEARDOWN"
  elif (( rc == 124 )); then
    verdict="TEARDOWN_HANG"
  elif (( newcores > 0 )); then
    verdict="TEARDOWN_SEGV"
  elif (( completed >= WANT_RANKS )) && (( rc == 0 )) && [[ "$ckpt" == "OK" ]]; then
    verdict="CLEAN"
  else
    verdict="OTHER_FAIL"
  fi

  echo "  -> $verdict  rc=$rc wall=${wall}s compile=${compile}s shutdown=${shutdown}s completed=$completed/$WANT_RANKS osexit_ranks=$exits cores+=$newcores surv=$surv_str ckpt=$ckpt"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$n" "$cond" "$eff_launcher" "$eff_opsync" "$shim" "$verdict" "$rc" "$wall" \
    "$compile" "$shutdown" "$completed" "$exits" "$newcores" "$surv_str" "$ckpt" >> "$RESULTS"

  # Reclaim the (capped) cores so counts stay per-trial and the tree stays clean.
  find "$CORE_DIR" -maxdepth 1 -name 'core.*' -newermt "@$t0" -delete 2>/dev/null
}

time_left() { echo $(( WALL_BUDGET - ( $(date +%s) - JOB_START ) )); }

# ===========================================================================
# PHASE 0 -- torchrun feasibility, from evidence already on disk. Costs no
# allocation time. Reported as a configuration finding, not as a trial.
# ===========================================================================
echo
echo "===== PHASE 0: torchrun feasibility ====="
TR_CCL=$(strings /opt/aurora/26.181.0/oneapi/ccl/2022.1/lib/libccl.so 2>/dev/null | grep -cix 'torchrun')
TR_CLI=$(grep -c 'torchrun' "$LAUNCH/src/cli/launch.py" 2>/dev/null)
TR_BIN=$(command -v torchrun 2>/dev/null || echo "<not on PATH>")
echo "  'torchrun' as a standalone token in libccl.so : ${TR_CCL:-0}  (0 = not a CCL_PROCESS_LAUNCHER value)"
echo "  'torchrun' mentions in src/cli/launch.py      : ${TR_CLI:-0}  (help text only; dispatch at :523 is mpiexec/mpirun/srun)"
echo "  torchrun binary                               : $TR_BIN"
echo "  VERDICT: CCL_PROCESS_LAUNCHER=torchrun is not a valid oneCCL value, and"
echo "           the launcher has no torchrun dispatch. NOT_IMPLEMENTED --"
echo "           a configuration gap, not a result about the runtime defect."

# ===========================================================================
# PHASE 1 -- pmix acceptance probe. Gates condition C.
# ===========================================================================
echo
echo "===== PHASE 1: pmix acceptance probe ====="
run_trial 1 "P_pmix_probe" pmix 1 on
PMIX_VERDICT=$(awk -F'\t' '$1=="1"{print $6}' "$RESULTS" | tail -1)
PMIX_EFF=$(awk -F'\t' '$1=="1"{print $3}' "$RESULTS" | tail -1)
if [[ "$PMIX_VERDICT" == "LAUNCHER_REJECTED" || "$PMIX_EFF" != "pmix" ]]; then
  PMIX_OK=0
  echo "  pmix NOT accepted (oneCCL reports launcher=$PMIX_EFF). Condition C dropped."
  echo "  This is a CONFIGURATION/COMPATIBILITY failure and is reported as such."
else
  PMIX_OK=1
  echo "  pmix accepted. Condition C enabled."
fi

# ===========================================================================
# PHASE 2 -- measurement. Round-robin so machine drift hits every condition
# equally. A first (positive control) -- if A never faults, nothing downstream
# of it means anything, and the report says so.
# ===========================================================================
CONDS=("A hydra 0 off" "B hydra 1 off" "S hydra 0 on")
(( PMIX_OK )) && CONDS+=("C pmix 1 off")

# PRIORITY ORDER, not plain round-robin. At ~505s per shim-OFF trial the hour
# holds only ~5-6 trials, so a naive rotation can leave the positive control
# at n=1 and the whole job uninterpretable. Front-load one A and one B (the
# control and the actual test), then rotate for replication with whatever
# time is left.
ORDER=("A hydra 0 off" "B hydra 1 off")
(( PMIX_OK )) && ORDER+=("C pmix 1 off")
ORDER+=("S hydra 0 on")
ORDER+=("${CONDS[@]}")

echo
echo "===== PHASE 2: measurement (${#CONDS[@]} conditions, control-first) ====="
echo "  trial timeout ${TRIAL_TIMEOUT}s, reserve ${RESERVE_S}s -> expect ~$(( ($(time_left) - RESERVE_S) / 620 + 1 )) more trials"
TRIAL=1
idx=0
while :; do
  left=$(time_left)
  if (( left < RESERVE_S )); then
    echo; echo "stopping: ${left}s walltime left, below the ${RESERVE_S}s reserved"
    echo "(a trial that started now could run ${TRIAL_TIMEOUT}s and be killed mid-report)"
    break
  fi
  spec="${ORDER[$(( idx % ${#ORDER[@]} ))]}"
  idx=$(( idx + 1 ))
  # shellcheck disable=SC2086
  set -- $spec
  TRIAL=$(( TRIAL + 1 ))
  run_trial "$TRIAL" "$1" "$2" "$3" "$4"
done

# ===========================================================================
# REPORT -- written BEFORE any expensive filesystem walk. The last job was
# killed mid-report at walltime because a `du` over Lustre ate the reserve.
# ===========================================================================
echo
echo "================= os._exit / CCL CONDITION REPORT ================="
column -t -s$'\t' "$RESULTS" 2>/dev/null || cat "$RESULTS"

echo
echo "--- per condition ---"
printf '%-6s %-8s %-8s %-5s %6s %6s %6s %6s %8s %8s %8s\n' \
  cond launcher op_sync shim n clean segv hang cores surv shutdown_s
for spec in "${CONDS[@]}"; do
  # shellcheck disable=SC2086
  set -- $spec
  awk -F'\t' -v c="$1" -v lab="$1" -v L="$2" -v O="$3" -v S="$4" '
    NR>1 && $2==c && $6!="ABORT_WARM_CACHE" {
      n++
      if ($6=="CLEAN") clean++
      if ($6=="TEARDOWN_SEGV") segv++
      if ($6=="TEARDOWN_HANG") hang++
      cores += ($13 ~ /^[0-9]+$/ ? $13 : 0)
      if ($14 ~ /UNREACH/) unreach=1; else surv += ($14 ~ /^[0-9]+$/ ? $14 : 0)
      if ($10 ~ /^[0-9]+$/) { sd += $10; sn++ }
    }
    END {
      printf "%-6s %-8s %-8s %-5s %6d %6d %6d %6d %8d %8s %8s\n",
        lab, L, O, S, n, clean, segv, hang, cores,
        (unreach ? "UNREACH" : surv),
        (sn ? sprintf("%.0f", sd/sn) : "-")
    }' "$RESULTS"
done

echo
echo "--- compile time (CLEAN trials only; isolation held ON everywhere) ---"
awk -F'\t' 'NR>1 && $6=="CLEAN" && $9 ~ /^[0-9]+$/ {s[$2]+=$9; n[$2]++}
  END {for (c in s) printf "  %-6s mean %.1fs over %d\n", c, s[c]/n[c], n[c]}' "$RESULTS"

echo
echo "--- VERDICT ---"
A_N=$(awk -F'\t' 'NR>1 && $2=="A" && $6!="ABORT_WARM_CACHE"' "$RESULTS" | wc -l)
A_BAD=$(awk -F'\t' 'NR>1 && $2=="A" && ($6=="TEARDOWN_SEGV"||$6=="TEARDOWN_HANG")' "$RESULTS" | wc -l)
B_N=$(awk -F'\t' 'NR>1 && $2=="B" && $6!="ABORT_WARM_CACHE"' "$RESULTS" | wc -l)
B_BAD=$(awk -F'\t' 'NR>1 && $2=="B" && ($6=="TEARDOWN_SEGV"||$6=="TEARDOWN_HANG")' "$RESULTS" | wc -l)
S_N=$(awk -F'\t' 'NR>1 && $2=="S" && $6!="ABORT_WARM_CACHE"' "$RESULTS" | wc -l)
S_CLEAN=$(awk -F'\t' 'NR>1 && $2=="S" && $6=="CLEAN"' "$RESULTS" | wc -l)

pw() { (( $1 < 2 )) && printf ' [UNDERPOWERED n<2]'; }
echo "  A (hydra, OP_SYNC unset, shim OFF) : $A_BAD/$A_N faulted   <- positive control$(pw "$A_N")"
echo "  B (hydra, OP_SYNC=1,     shim OFF) : $B_BAD/$B_N faulted   <- test 1$(pw "$B_N")"
(( PMIX_OK )) && {
  C_N=$(awk -F'\t' 'NR>1 && $2=="C" && $6!="ABORT_WARM_CACHE"' "$RESULTS" | wc -l)
  C_BAD=$(awk -F'\t' 'NR>1 && $2=="C" && ($6=="TEARDOWN_SEGV"||$6=="TEARDOWN_HANG")' "$RESULTS" | wc -l)
  echo "  C (pmix,  OP_SYNC=1,     shim OFF) : $C_BAD/$C_N faulted   <- test 2"
} || echo "  C (pmix)                           : NOT RUN -- oneCCL rejected the launcher value (config failure)"
echo "  S (shim ON, safety control)        : $S_CLEAN/$S_N clean"
echo
if (( A_N > 0 && A_BAD == 0 )); then
  echo "  !! POSITIVE CONTROL DID NOT REPRODUCE. Condition A was expected to fault"
  echo "     with the shim off and did not. Every other arm is therefore"
  echo "     UNINTERPRETABLE: a clean B or C says nothing when the baseline is"
  echo "     also clean. Do not conclude os._exit() is unnecessary from this job."
elif (( A_N == 0 )); then
  echo "  !! No condition-A trials completed. Nothing is interpretable."
else
  echo "  Positive control reproduced ($A_BAD/$A_N). B and C are interpretable"
  echo "  against it; see the per-condition table for rates."
  if (( B_N > 0 && B_BAD == 0 )); then
    echo
    echo "  B was clean in $B_N/$B_N trials while A faulted. That is consistent with"
    echo "  CCL_OP_SYNC=1 removing the need for os._exit(), but at n=$B_N it is a"
    echo "  LEAD, not a conclusion: this fault is deterministic per run, yet the"
    echo "  10-08 config change is only correlated with its first appearance."
    echo "  Confirm with a longer campaign before dropping the shim."
  fi
fi
echo
echo "  Arms with n<2 are marked UNDERPOWERED above and must not be read as"
echo "  results. This fault is deterministic (48/48 ranks, same gdb stack), so"
echo "  a handful of trials distinguishes presence from absence -- but only"
echo "  once the positive control has actually fired."
echo
echo "  NOTE: cores were capped at 2 MB in-process (AGPT_CORE_LIMIT_BYTES, applied"
echo "  by the shim because PALS does not propagate rlimits) and deleted per trial."
echo "  Counts are occurrence counts, not sizes. Shim-OFF arms would otherwise have"
echo "  written ~542 GB per trial, inside the teardown window being measured."
CAP_RANKS=$(grep -ho 'INDUCTOR_ISO rank=[0-9]* core limit ->' "$LOGDIR"/trial*.log 2>/dev/null \
            | awk '{print $2}' | sort -u | wc -l)
echo "  distinct ranks confirming the cap across all trials: $CAP_RANKS (expect $WANT_RANKS)"

echo
echo "cache usage (last, cheap):"
timeout 60 du -sm "$CACHE_ROOT" 2>/dev/null | awk '{printf "  %s MB total\n", $1}' || echo "  (du timed out; skipped)"
echo
echo "=== done at $(date) ==="
echo "logs: $LOGDIR"
