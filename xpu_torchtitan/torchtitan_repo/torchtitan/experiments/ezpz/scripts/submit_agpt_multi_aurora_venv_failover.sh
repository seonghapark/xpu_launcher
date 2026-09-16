#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-multi-failover
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
# select= overridden via qsub -l select=N. Default production layout:
#   1536 train (512+512+256+256) + 4*SPARES. SPARES=10 -> select=1576.
#PBS -q prod
#PBS -j oe

# ============================================================================
# Umbrella job: advance ALL 4 canonical AGPT chains in ONE PBS allocation.
# ============================================================================
#
# Why this exists
# ---------------
# The 512N canonical chains (2B + 20B) starve for weeks in the `small` queue:
# a 522-node scatter request rarely wins a slot against the crowd. Packing all
# four canonical chains (2B-512, 20B-512, 2B-256, 20B-256) into ONE job of
# ~1536 train nodes routes to the far-less-contended `medium` queue (1025-1999
# node band) and advances every chain in a single dispatch instead of having
# them compete for four separate scarce slots.
#
# How it works
# ------------
# This is a THIN orchestrator. It does NOT reimplement any training or failover
# logic. It:
#   1. Slices the flat $PBS_NODEFILE into 4 DISJOINT hostfiles (one per chain).
#   2. Launches the existing per-chain submit scripts
#      (submit_agpt_2b_aurora_venv_failover.sh / submit_agpt_20b_...) as 4
#      concurrent background subprocesses, each with a fully ISOLATED env block.
#   3. wait()s on all 4 and reports per-trainer exit codes. One chain crashing
#      does NOT abort the others.
#
# Each child is the clone's OWN copy of the battle-tested submit script -- the
# exact code the canonical chains already run -- driven entirely via exported
# environment. No edits to failover_lib.sh, the child scripts, or core
# torchtitan are required.
#
# Per-trainer isolation (all four share ONE $PBS_JOBID + ONE allocation)
# ---------------------------------------------------------------------
#   PBS_NODEFILE      -> this chain's disjoint slice (failover_init reads ONLY
#                        this file, never /var/spool/pbs/aux).
#   FAILOVER_LOG_DIR  -> unique per trainer ("-t$i"). The default keys on
#                        $PBS_JOBID, which is IDENTICAL across all 4; trainers 0
#                        and 2 ALSO share a clone + cwd, so the suffix is
#                        load-bearing. active/spare/bad hostfiles + attempt logs
#                        all derive from this dir.
#   QUEUE_WAIT_CSV    -> unique per trainer (default is cwd-relative, shared by
#                        the two 2B trainers otherwise).
#   MASTER_ADDR       -> UNSET. Each child's independent `ezpz launch` derives
#                        its own rank-0 master_addr (with the .hsn.cm... HSN
#                        suffix) from its OWN first host. Pinning a bare name
#                        would break HSN routing; inheriting a stale value would
#                        collide -- so we unset it.
#   MASTER_PORT       -> distinct per trainer (defensive; respected by ezpz's
#                        rank-0 path). Disjoint slices already isolate the
#                        rendezvous host, so this is belt-and-suspenders.
#   PBS_O_WORKDIR     -> this chain's clone dir (drives the child's cd,
#                        SCRIPTS_DIR, relative .venv / checkpoints / data-lists).
#   CKPT_DIR          -> this chain's checkpoint dir (explicit, so the two 2B
#                        trainers in the same clone never collide).
#
# Usage
# -----
#   # Production (real 1576-node submit):
#   qsub -q prod -A AuroraGPT -l select=1576 -l walltime=12:00:00 \
#     -l filesystems=home:flare -N agpt-multi-failover -j oe \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_multi_aurora_venv_failover.sh
#
#   # Layer 1 -- pure dry-run on a login node (no allocation, zero cost):
#   DRY_RUN=1 PBS_JOBID=9999999 PBS_O_WORKDIR=$PWD \
#     PBS_NODEFILE=/path/to/fixture-1576-hosts.txt \
#     bash torchtitan/experiments/ezpz/scripts/submit_agpt_multi_aurora_venv_failover.sh
#
#   # Layer 2 -- tiny real concurrency (debug-scaling, <=256 nodes):
#   qsub -q debug-scaling -A AuroraGPT -l select=8 -l walltime=01:00:00 \
#     -l filesystems=home:flare -N agpt-multi-smoke -j oe \
#     -v MULTI_PROFILE=tiny \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_multi_aurora_venv_failover.sh
#
# Env knobs
# ---------
#   MULTI_PROFILE   prod (default) | tiny  -- tiny shrinks every chain to
#                   MULTI_TINY_NNODES nodes, runs MULTI_TINY_STEPS steps, and
#                   writes to THROWAWAY ckpt dirs (never the canonical chains).
#   SPARES          spare nodes per chain for failover (default 10).
#   DRY_RUN         1 -> slice + print resolved env blocks, then exit (no launch).
#   LAUNCH_STAGGER  seconds between background launches (default 30) so the 4
#                   venv broadcasts + palsd-kills don't fire in the same instant.
#   FAILOVER_MAX_RETRIES  passed through to each child (default 3).
#   MULTI_TINY_NNODES (default 2), MULTI_TINY_SPARES (default 0),
#   MULTI_TINY_STEPS (default 3)  -- tiny-profile sizing.
#
# NOTE: deliberately NO `set -euo pipefail`. The children manage their own
# error handling and rely on unbound-var-tolerant venv activation; the umbrella
# must survive a single child failing without aborting the rest.
# ============================================================================

RUNS="/flare/AuroraGPT/foremans/runs"

PROFILE="${MULTI_PROFILE:-prod}"
SPARES="${SPARES:-10}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-30}"
TINY_NNODES="${MULTI_TINY_NNODES:-2}"
TINY_SPARES="${MULTI_TINY_SPARES:-0}"
TINY_STEPS="${MULTI_TINY_STEPS:-3}"
JOBID="${PBS_JOBID%%.*}"
[[ -z "$JOBID" ]] && JOBID="nojob"

# ---- Per-trainer config table -------------------------------------------------
# Format: model|nnodes|master_port|workdir|ckpt_dir(relative; torchtitan adds
# the outputs/ prefix). GBS is NOT set here -- the child computes it from the
# slice as NGPUS*LBS(2)*GAS(1): 512*12*2=12288, 256*12*2=6144.
TRAINERS=(
    "2b|512|29500|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-sophiag-olmo-mix-1124-n512-gbs12288"
    "20b|512|29600|$RUNS/agpt-20b-v2/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs12288"
    "2b|256|29700|$RUNS/agpt-2b-v2/torchtitan-ezpz|checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144"
    "20b|256|29800|$RUNS/agpt-20b-n256/torchtitan-ezpz|checkpoints/agpt-20b-sophiag-olmo-mix-1124-n256-gbs6144"
)

log() { echo "[multi $(date +%H:%M:%S)] $*"; }
die() { echo "[multi ERROR] $*" >&2; exit 1; }

# ---- Validate allocation ------------------------------------------------------
[[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] \
    || die "PBS_NODEFILE not set or missing (got '${PBS_NODEFILE:-}')"

TOTAL_AVAIL=$(wc -l < "$PBS_NODEFILE")

# Compute the required node count from the (possibly tiny-overridden) table.
need=0
for row in "${TRAINERS[@]}"; do
    IFS='|' read -r _m nnodes _p _w _c <<< "$row"
    if [[ "$PROFILE" == "tiny" ]]; then
        need=$(( need + TINY_NNODES + TINY_SPARES ))
    else
        need=$(( need + nnodes + SPARES ))
    fi
done

log "profile=$PROFILE  spares/chain=$([[ $PROFILE == tiny ]] && echo $TINY_SPARES || echo $SPARES)"
log "nodes: need=$need  available=$TOTAL_AVAIL  (jobid=$JOBID)"
if (( TOTAL_AVAIL < need )); then
    if (( DRY_RUN == 1 )); then
        log "WARNING (dry-run): allocation $TOTAL_AVAIL < need $need; slices will be short"
    else
        die "allocation too small: have $TOTAL_AVAIL nodes, need $need"
    fi
fi

# ---- Slice dir (ABSOLUTE so children resolve it after cd-ing to their clone) --
MULTI_LOG_DIR="${MULTI_LOG_DIR:-${PBS_O_WORKDIR:-$PWD}/logs/multi-${JOBID}}"
mkdir -p "$MULTI_LOG_DIR" || die "cannot create $MULTI_LOG_DIR"
case "$MULTI_LOG_DIR" in
    /*) : ;;
    *)  MULTI_LOG_DIR="$(cd "$MULTI_LOG_DIR" && pwd)" ;;
esac
log "slice + console logs under: $MULTI_LOG_DIR"

# ---- Slice the nodefile + build each trainer's resolved plan ------------------
declare -a T_MODEL T_NNODES T_PORT T_WORKDIR T_CKPT T_SLICE T_CHILD T_MASTER
declare -a T_VENVDST T_VENVSRC
offset=1
for idx in "${!TRAINERS[@]}"; do
    IFS='|' read -r model nnodes port workdir ckpt <<< "${TRAINERS[$idx]}"

    # child_model selects which submit script runs; it equals `model` in prod.
    # In tiny profile every trainer runs the 2B child (the 20B model will not
    # fit on TINY_NNODES nodes, and the 20B child path is byte-identical to the
    # 2B one apart from the MODEL string). The original `model` label is kept
    # only for the throwaway ckpt-dir name so the 4 smoke dirs stay distinct.
    child_model="$model"
    if [[ "$PROFILE" == "tiny" ]]; then
        child_model="2b"
        nnodes="$TINY_NNODES"
        spares="$TINY_SPARES"
        # Throwaway ckpt dir -- NEVER the canonical chain. Keyed by jobid+idx.
        ckpt="checkpoints/multi-smoke-${JOBID}/t${idx}-${model}-n${nnodes}"
    else
        spares="$SPARES"
    fi

    slice_count=$(( nnodes + spares ))
    slice_file="$MULTI_LOG_DIR/trainer-${idx}.hostfile"
    sed -n "${offset},$((offset + slice_count - 1))p" "$PBS_NODEFILE" > "$slice_file"
    got=$(wc -l < "$slice_file")
    if (( got != slice_count && DRY_RUN != 1 )); then
        die "trainer $idx: sliced $got nodes, expected $slice_count (offset=$offset)"
    fi
    master_addr=$(head -n 1 "$slice_file")
    offset=$(( offset + slice_count ))

    child="$workdir/torchtitan/experiments/ezpz/scripts/submit_agpt_${child_model}_aurora_venv_failover.sh"

    T_MODEL[$idx]="$model"
    T_NNODES[$idx]="$nnodes"
    T_PORT[$idx]="$port"
    T_WORKDIR[$idx]="$workdir"
    T_CKPT[$idx]="$ckpt"
    T_SLICE[$idx]="$slice_file"
    T_CHILD[$idx]="$child"
    T_MASTER[$idx]="$master_addr"
    # Per-model venv dst on each node. Two trainers of the same model run on
    # DISJOINT slices, so a shared dst name is safe (it resolves to physically
    # distinct node-local dirs); only the mom node is shared, and distinct
    # per-model names keep its 2b and 20b copies from clobbering each other.
    T_VENVDST[$idx]="/tmp/.venv-${child_model}"
    # Resolve symlinks so two clones that share one tarball (e.g. 20b-n256's
    # .venv.tar.gz symlinks to 20b-v2's) collapse to ONE pre-stage broadcast.
    _vsrc="$workdir/.venv.tar.gz"
    [[ -e "$_vsrc" ]] && _vsrc="$(readlink -f "$_vsrc")"
    T_VENVSRC[$idx]="$_vsrc"
done

# ---- Disjointness assertion (no node may appear in two slices) ---------------
dupes=$(cat "$MULTI_LOG_DIR"/trainer-*.hostfile 2>/dev/null | sort | uniq -d)
if [[ -n "$dupes" ]]; then
    die "slices overlap -- the same node(s) appear in multiple trainers:"$'\n'"$dupes"
fi
log "slices are disjoint (no shared nodes)"

# ---- Existence checks ---------------------------------------------------------
for idx in "${!TRAINERS[@]}"; do
    [[ -d "${T_WORKDIR[$idx]}" ]] || die "trainer $idx workdir missing: ${T_WORKDIR[$idx]}"
    [[ -f "${T_CHILD[$idx]}"   ]] || die "trainer $idx child script missing: ${T_CHILD[$idx]}"
done

# ---- Resolved-plan banner -----------------------------------------------------
echo "============================================================"
echo "  AGPT multi-chain umbrella -- resolved plan"
echo "============================================================"
for idx in "${!TRAINERS[@]}"; do
    printf "  trainer %s: %-3s  n=%-4s spare=%s  port=%s\n" \
        "$idx" "${T_MODEL[$idx]}" "${T_NNODES[$idx]}" \
        "$([[ $PROFILE == tiny ]] && echo $TINY_SPARES || echo $SPARES)" "${T_PORT[$idx]}"
    printf "             workdir : %s\n" "${T_WORKDIR[$idx]}"
    printf "             ckpt    : %s\n" "${T_CKPT[$idx]}"
    printf "             slice   : %s (%s nodes, master=%s)\n" \
        "${T_SLICE[$idx]}" "$(wc -l < "${T_SLICE[$idx]}")" "${T_MASTER[$idx]}"
    printf "             child   : %s\n" "${T_CHILD[$idx]}"
done
echo "============================================================"

# ---- Pre-stage venvs (ONE broadcast per distinct venv, serially) -------------
# Each child runs on the shared mom node, and `ezpz yeet` always does its
# "local copy" on the mom node at a fixed dst; 4 concurrent child yeets would
# corrupt each other's /tmp copy (Layer-2 smoke 8566938: all rc=127). So the
# umbrella broadcasts each DISTINCT venv exactly once, serially, to the union
# of the nodes that need it, each to its own per-model dst. Children then run
# with SKIP_YEET=1 + VENV_DST pointing at the pre-staged copy.
#
# Dedup key = "src|dst". For each unique pair, the target hostfile is the
# concatenation of every trainer slice that uses it (active + spare), since the
# child's failover_swap_in may promote a spare and expects the venv present
# there too.
prestage_venvs() {
    local -A seen=()        # "src|dst" -> 1 once staged
    local idx src dst key hostfile rc
    # Activate the umbrella's own repo venv so `ezpz` is on PATH. Use trainer
    # 0's clone (any works; they all have a usable .venv for the CLI).
    if [[ -f "${T_WORKDIR[0]}/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${T_WORKDIR[0]}/.venv/bin/activate" || die "cannot activate ${T_WORKDIR[0]}/.venv for ezpz CLI"
    fi
    for idx in "${!TRAINERS[@]}"; do
        src="${T_VENVSRC[$idx]}"
        dst="${T_VENVDST[$idx]}"
        key="${src}|${dst}"
        [[ -n "${seen[$key]:-}" ]] && continue
        seen[$key]=1
        # Build the union hostfile for this (src,dst): every slice with the
        # same dst (same model) contributes its nodes.
        hostfile="$MULTI_LOG_DIR/prestage-$(basename "$dst").hostfile"
        : > "$hostfile"
        local j
        for j in "${!TRAINERS[@]}"; do
            if [[ "${T_VENVSRC[$j]}|${T_VENVDST[$j]}" == "$key" ]]; then
                cat "${T_SLICE[$j]}" >> "$hostfile"
            fi
        done
        sort -u "$hostfile" -o "$hostfile"
        [[ -f "$src" ]] || die "prestage: venv tarball missing: $src"
        log "prestage: $src -> $dst on $(wc -l < "$hostfile") nodes"
        ezpz yeet-env --src "$src" --hostfile "$hostfile" --dst "$dst"
        rc=$?
        (( rc == 0 )) || die "prestage failed (rc=$rc) for $src -> $dst"
        log "prestage OK: $dst"
    done
    [[ -n "${VIRTUAL_ENV:-}" ]] && deactivate 2>/dev/null
    return 0
}

# ---- Launch one trainer in an isolated background subshell --------------------
# Backgrounds directly in the CALLER's shell and records the pid into PIDS[idx].
# It must NOT be called via command substitution ($(...)): that would run the
# `&` in a subshell, making the job a child of THAT subshell, so the main
# shell's later `wait "${PIDS[idx]}"` would fail with "pid is not a child of
# this shell" and return 127 for every trainer (observed: smoke 8567358, all
# 4 spuriously rc=127 the instant they launched, while they were really still
# starting up). Call it plainly: `launch_trainer "$idx"`.
launch_trainer() {
    local idx="$1"
    local console="$MULTI_LOG_DIR/trainer-${idx}.console.log"
    (
        # --- isolated env (see header for the rationale of each) ---
        export PBS_O_WORKDIR="${T_WORKDIR[$idx]}"
        export PBS_NODEFILE="${T_SLICE[$idx]}"
        export NHOSTS_TRAIN="${T_NNODES[$idx]}"
        export CKPT_DIR="${T_CKPT[$idx]}"
        export FAILOVER_LOG_DIR="${T_WORKDIR[$idx]}/logs/failover-${JOBID}-t${idx}"
        export QUEUE_WAIT_CSV="${T_WORKDIR[$idx]}/logs/queue_wait-multi-${JOBID}-t${idx}.csv"
        export FAILOVER_MAX_RETRIES="${FAILOVER_MAX_RETRIES:-3}"
        # Venv was pre-staged by the umbrella (prestage_venvs); the child must
        # NOT run its own yeet (would race on the shared mom node) and must
        # activate the per-model pre-staged copy.
        export SKIP_YEET=1
        export VENV_DST="${T_VENVDST[$idx]}"
        # Rendezvous: distinct port; addr unset so each child derives its own
        # rank-0 HSN address from its own slice.
        unset MASTER_ADDR
        export MASTER_PORT="${T_PORT[$idx]}"
        # Tiny profile: cap steps so the smoke test finishes fast.
        if [[ "$PROFILE" == "tiny" ]]; then
            export TRAINING_STEPS="$TINY_STEPS"
        fi
        cd "${T_WORKDIR[$idx]}" || exit 97
        # --login so the child's `module load` works (module is a shell
        # function not inherited by a plain `bash child.sh`).
        exec bash --login "${T_CHILD[$idx]}"
    ) > "$console" 2>&1 &
    # Record the pid in the caller's shell so `wait` recognizes it as a child.
    PIDS[$idx]=$!
}

if (( DRY_RUN == 1 )); then
    echo
    log "DRY_RUN=1 -- env blocks that WOULD be exported per trainer:"
    for idx in "${!TRAINERS[@]}"; do
        echo "  --- trainer $idx (${T_MODEL[$idx]}, n=${T_NNODES[$idx]}) ---"
        echo "      PBS_O_WORKDIR   = ${T_WORKDIR[$idx]}"
        echo "      PBS_NODEFILE    = ${T_SLICE[$idx]}"
        echo "      NHOSTS_TRAIN    = ${T_NNODES[$idx]}"
        echo "      CKPT_DIR        = ${T_CKPT[$idx]}"
        echo "      FAILOVER_LOG_DIR= ${T_WORKDIR[$idx]}/logs/failover-${JOBID}-t${idx}"
        echo "      QUEUE_WAIT_CSV  = ${T_WORKDIR[$idx]}/logs/queue_wait-multi-${JOBID}-t${idx}.csv"
        echo "      MASTER_ADDR     = (unset -- child derives from slice head ${T_MASTER[$idx]})"
        echo "      MASTER_PORT     = ${T_PORT[$idx]}"
        echo "      SKIP_YEET       = 1"
        echo "      VENV_DST        = ${T_VENVDST[$idx]}  (src ${T_VENVSRC[$idx]})"
        echo "      child           = bash --login ${T_CHILD[$idx]}"
    done
    echo
    log "DRY_RUN: pre-stage broadcasts that WOULD run (one per distinct src|dst):"
    declare -A _seen=()
    for idx in "${!TRAINERS[@]}"; do
        k="${T_VENVSRC[$idx]}|${T_VENVDST[$idx]}"
        [[ -n "${_seen[$k]:-}" ]] && continue
        _seen[$k]=1
        n=0; for j in "${!TRAINERS[@]}"; do [[ "${T_VENVSRC[$j]}|${T_VENVDST[$j]}" == "$k" ]] && n=$(( n + $(wc -l < "${T_SLICE[$j]}") )); done
        echo "      ezpz yeet-env --src ${T_VENVSRC[$idx]} --dst ${T_VENVDST[$idx]} (~$n nodes)"
    done
    echo
    log "DRY_RUN complete -- nothing launched."
    exit 0
fi

# ---- Pre-stage all distinct venvs BEFORE launching any child -----------------
prestage_venvs

# ---- Launch all 4, staggered --------------------------------------------------
declare -a PIDS
for idx in "${!TRAINERS[@]}"; do
    launch_trainer "$idx"   # sets PIDS[$idx] directly (NOT via $(...) -- see note)
    log "launched trainer $idx (${T_MODEL[$idx]} n=${T_NNODES[$idx]}) pid=${PIDS[$idx]} -> $MULTI_LOG_DIR/trainer-${idx}.console.log"
    # Stagger all but the last.
    if (( idx < ${#TRAINERS[@]} - 1 )); then
        sleep "$LAUNCH_STAGGER"
    fi
done

log "all ${#TRAINERS[@]} trainers launched; waiting for completion..."

# ---- Wait + capture per-trainer rc (failure isolation) -----------------------
declare -a RC
fails=0
for idx in "${!TRAINERS[@]}"; do
    wait "${PIDS[$idx]}"
    RC[$idx]=$?
    (( RC[$idx] != 0 )) && fails=$(( fails + 1 ))
    log "trainer $idx (${T_MODEL[$idx]} n=${T_NNODES[$idx]}) finished rc=${RC[$idx]}"
done

# ---- Summary ------------------------------------------------------------------
echo "============================================================"
echo "  AGPT multi-chain umbrella -- summary (jobid=$JOBID)"
echo "============================================================"
for idx in "${!TRAINERS[@]}"; do
    status=$([[ ${RC[$idx]} -eq 0 ]] && echo OK || echo "FAIL(${RC[$idx]})")
    printf "  trainer %s: %-3s n=%-4s  %-9s  %s\n" \
        "$idx" "${T_MODEL[$idx]}" "${T_NNODES[$idx]}" "$status" \
        "$MULTI_LOG_DIR/trainer-${idx}.console.log"
done
echo "  failed: $fails / ${#TRAINERS[@]}"
echo "============================================================"

exit "$fails"
