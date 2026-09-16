#!/bin/bash
# failover_lib.sh — bad-node failover support for ezpz/torchtitan training.
#
# Source this file from a submit script that has requested
# NHOSTS_TRAIN + NHOSTS_SPARE nodes via PBS. Then call:
#
#   failover_init NHOSTS_TRAIN
#   failover_run <command...>
#
# `failover_init` splits PBS_NODEFILE → active.hostfile (first N lines)
# + spare.hostfile (rest), exports PBS_NODEFILE → active.hostfile so
# downstream `ezpz launch` only sees the training subset, and yeets the
# venv to ALL nodes (active + spare) so any spare can swap in cleanly.
#
# `failover_run` runs the command, scrapes the output for known
# bad-node failure modes, swaps the offending node out for a spare,
# and retries (up to FAILOVER_MAX_RETRIES, default 3).
#
# Detected failure modes (see scrape_bad_nodes.py):
#  - `<hostname>: rank N died from signal {9,11,15}` (PALS shepherd kill)
#  - `RuntimeError: ... Connection closed by peer [IP]:port` (gloo TCP)
#  - `RuntimeError: ... Timed out waiting Nms for recv` (gloo timeout)
#  - `MemoryError: std::bad_alloc` in `torch.distributed.broadcast` (init OOM —
#    no specific node, rotates 1 spare in blindly)
#
# Env vars (all optional):
#   FAILOVER_MAX_RETRIES   default 3
#   FAILOVER_LOG_DIR       default $(pwd)/logs/failover-${PBS_JOBID%%.*}
#   FAILOVER_KEEP_BAD      default 1 — append bad node to bad_nodes.txt for inspection

set -o pipefail  # don't `set -euo pipefail` here — venv activate has unbound vars

_failover_log() { echo "[failover] $*" >&2; }

# ---------------------------------------------------------------------------
# failover_init NHOSTS_TRAIN
#
# Splits PBS_NODEFILE into active + spare files in $FAILOVER_LOG_DIR.
# Exports PBS_NODEFILE → active.hostfile.
# Yeets venv to ALL nodes (so spare swap-in is instantaneous later).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# failover_log_queue_wait
#
# Record this job's queue-wait (qtime -> now/start) to a shared CSV so we
# have a precise, durable record after the job ages out of `qstat`. PBS
# does not pass qtime as an env var, so we read it back from the job's own
# qstat record (available while the job runs). Best-effort: never fails the
# job. Columns: jobid,jobname,nodes,queue,qtime,stime,wait_seconds,wait_h
# ---------------------------------------------------------------------------
failover_log_queue_wait() {
    local jobid="${PBS_JOBID%%.*}"
    [[ -n "$jobid" ]] || return 0
    local csv="${QUEUE_WAIT_CSV:-${PBS_O_WORKDIR:-$PWD}/logs/queue_wait.csv}"
    mkdir -p "$(dirname "$csv")" 2>/dev/null || return 0
    local rec qt jn nodes q
    rec=$(qstat -xf "$PBS_JOBID" 2>/dev/null) || return 0
    qt=$(echo "$rec"  | grep -m1 "qtime "    | sed 's/.*= //')
    jn=$(echo "$rec"  | grep -m1 "Job_Name"  | sed 's/.*= //')
    nodes=$(echo "$rec" | grep -m1 "Resource_List.nodect" | sed 's/.*= //')
    q=$(echo "$rec"   | grep -m1 "queue ="   | sed 's/.*= //')
    local stt qts sts wait
    # Prefer the job's actual stime (start). Falls back to "now" -- which
    # is correct when this runs at job start (now ~= stime), and is the
    # only option if stime isn't published yet.
    stt=$(echo "$rec"  | grep -m1 "stime "    | sed 's/.*= //')
    qts=$(date -d "$qt" +%s 2>/dev/null) || return 0
    sts=$(date -d "$stt" +%s 2>/dev/null) || sts=$(date +%s)
    wait=$(( sts - qts ))
    [[ -f "$csv" ]] || echo "jobid,jobname,nodes,queue,qtime,stime,wait_seconds,wait_hours" > "$csv"
    printf '%s,%s,%s,%s,%s,%s,%d,%.1f\n' \
        "$jobid" "$jn" "$nodes" "$q" \
        "$(date -d "@$qts" -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)" \
        "$(date -d "@$sts" -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null)" \
        "$wait" "$(awk "BEGIN{print $wait/3600}")" >> "$csv"
    _failover_log "queue wait: $(awk "BEGIN{printf \"%.1f\", $wait/3600}")h (qtime->start); logged to $csv"
}

# ---------------------------------------------------------------------------
# failover_remaining_walltime
#
# Echo the number of seconds of PBS walltime REMAINING for this job, for use
# as WALLTIME_SECONDS by the trainer's walltime-aware checkpointing. PBS does
# not export walltime as an env var, so we read it back from the job's own
# qstat record: remaining = Resource_List.walltime - (now - stime).
#
# "Remaining" (not total) is what the trainer wants: it starts its budget
# clock at loop entry, AFTER yeet/preflight/ckpt-load have already burned
# part of the wall, so subtracting elapsed-since-start here keeps the two in
# sync. Best-effort: echoes 0 (disables the feature) if anything is missing.
# ---------------------------------------------------------------------------
failover_remaining_walltime() {
    [[ -n "${PBS_JOBID:-}" ]] || { echo 0; return 0; }
    local rec wt stt total now_s start_s elapsed remain
    rec=$(qstat -xf "$PBS_JOBID" 2>/dev/null) || { echo 0; return 0; }
    wt=$(echo "$rec"  | grep -m1 "Resource_List.walltime" | sed 's/.*= //')
    stt=$(echo "$rec" | grep -m1 "stime "                 | sed 's/.*= //')
    [[ -n "$wt" ]] || { echo 0; return 0; }
    # Walltime is HH:MM:SS (or D+HH:MM:SS). Convert to seconds.
    total=$(awk -F: 'NF==3{print ($1*3600)+($2*60)+$3} NF==2{print ($1*60)+$2} NF==1{print $1}' <<< "$wt")
    [[ -n "$total" && "$total" -gt 0 ]] 2>/dev/null || { echo 0; return 0; }
    now_s=$(date +%s)
    start_s=$(date -d "$stt" +%s 2>/dev/null) || start_s="$now_s"
    elapsed=$(( now_s - start_s ))
    (( elapsed < 0 )) && elapsed=0
    remain=$(( total - elapsed ))
    (( remain < 0 )) && remain=0
    echo "$remain"
}

# ---------------------------------------------------------------------------
# failover_deadline_epoch
#
# Echo the ABSOLUTE Unix timestamp (job_start + walltime) at which PBS will
# kill this job, for use as WALLTIME_DEADLINE_EPOCH by the trainer's
# walltime-aware checkpointing. Unlike failover_remaining_walltime (a relative
# budget that resets every time the trainer relaunches on a failover retry),
# this absolute deadline is STABLE across retries -- so the backstop still
# fires before the real PBS walltime even after a mid-job bad-node swap
# (fixes the 2026-06-27 retry-resets-budget bug). Best-effort: echoes 0
# (disables) if anything is missing.
# ---------------------------------------------------------------------------
failover_deadline_epoch() {
    [[ -n "${PBS_JOBID:-}" ]] || { echo 0; return 0; }
    local rec wt stt total start_s
    rec=$(qstat -xf "$PBS_JOBID" 2>/dev/null) || { echo 0; return 0; }
    wt=$(echo "$rec"  | grep -m1 "Resource_List.walltime" | sed 's/.*= //')
    stt=$(echo "$rec" | grep -m1 "stime "                 | sed 's/.*= //')
    [[ -n "$wt" ]] || { echo 0; return 0; }
    total=$(awk -F: 'NF==3{print ($1*3600)+($2*60)+$3} NF==2{print ($1*60)+$2} NF==1{print $1}' <<< "$wt")
    [[ -n "$total" && "$total" -gt 0 ]] 2>/dev/null || { echo 0; return 0; }
    # Prefer the real job start (stime); fall back to now if not yet published.
    start_s=$(date -d "$stt" +%s 2>/dev/null) || start_s=$(date +%s)
    echo "$(( start_s + total ))"
}

failover_init() {
    local nhosts_train="$1"
    [[ -n "$nhosts_train" ]] || { _failover_log "ERROR: failover_init needs NHOSTS_TRAIN arg"; return 1; }
    [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]] || { _failover_log "ERROR: PBS_NODEFILE not set or missing"; return 1; }

    local total
    total=$(wc -l < "$PBS_NODEFILE")
    if (( total < nhosts_train )); then
        _failover_log "ERROR: PBS gave us $total nodes, training needs $nhosts_train"
        return 1
    fi

    export FAILOVER_LOG_DIR="${FAILOVER_LOG_DIR:-$(pwd)/logs/failover-${PBS_JOBID%%.*}}"
    mkdir -p "$FAILOVER_LOG_DIR"

    export FAILOVER_PBS_NODEFILE_ORIG="$PBS_NODEFILE"
    export FAILOVER_ACTIVE="$FAILOVER_LOG_DIR/active.hostfile"
    export FAILOVER_SPARE="$FAILOVER_LOG_DIR/spare.hostfile"
    export FAILOVER_BAD="$FAILOVER_LOG_DIR/bad_nodes.txt"

    head -n "$nhosts_train"   "$PBS_NODEFILE" > "$FAILOVER_ACTIVE"
    tail -n +"$((nhosts_train + 1))" "$PBS_NODEFILE" > "$FAILOVER_SPARE"
    : > "$FAILOVER_BAD"  # truncate

    export PBS_NODEFILE="$FAILOVER_ACTIVE"
    export NHOSTS="$nhosts_train"

    _failover_log "PBS gave us $total nodes ($(wc -l < "$FAILOVER_ACTIVE") active, $(wc -l < "$FAILOVER_SPARE") spare)"
    _failover_log "active hostfile: $FAILOVER_ACTIVE"
    _failover_log "spare hostfile:  $FAILOVER_SPARE"
    _failover_log "bad-node log:    $FAILOVER_BAD"

    # Record how long this job waited in the queue (best-effort).
    failover_log_queue_wait
}

# ---------------------------------------------------------------------------
# failover_yeet_all
#
# Run `ezpz yeet-env --src .venv.tar.gz` against the FULL nodelist
# (active + spare), so a spare can swap in without re-broadcast.
# Restores PBS_NODEFILE → active afterwards.
# ---------------------------------------------------------------------------
failover_yeet_all() {
    local full_nodefile="$FAILOVER_LOG_DIR/all.hostfile"
    cat "$FAILOVER_ACTIVE" "$FAILOVER_SPARE" > "$full_nodefile"
    local saved_pbs_nodefile="$PBS_NODEFILE"
    local saved_nhosts="$NHOSTS"
    export PBS_NODEFILE="$full_nodefile"
    export NHOSTS=$(wc -l < "$full_nodefile")
    _failover_log "yeet-env to ALL $NHOSTS nodes (active + spare)"
    if [[ -f .venv.tar.gz ]]; then
        ezpz yeet-env --src .venv.tar.gz
    else
        ezpz yeet-env
    fi
    local rc=$?
    export PBS_NODEFILE="$saved_pbs_nodefile"
    export NHOSTS="$saved_nhosts"
    return $rc
}

# ---------------------------------------------------------------------------
# failover_swap_in BAD_HOSTNAMES...
#
# For each bad hostname, remove it from active.hostfile and replace it
# with the next available spare (popping from spare.hostfile). Logs
# all bad hostnames to bad_nodes.txt for postmortem.
# Returns 0 if all swaps succeeded, 1 if we ran out of spares.
# ---------------------------------------------------------------------------
failover_swap_in() {
    local bad
    local swapped=0
    for bad in "$@"; do
        # Bad node must actually be in active set
        if ! grep -qxF "$bad" "$FAILOVER_ACTIVE" 2>/dev/null; then
            _failover_log "skip: $bad not in active hostfile"
            continue
        fi
        # Need a spare
        local spare
        spare=$(head -n 1 "$FAILOVER_SPARE")
        if [[ -z "$spare" ]]; then
            _failover_log "ERROR: out of spares — cannot replace $bad"
            return 1
        fi
        # Pop spare, swap in
        tail -n +2 "$FAILOVER_SPARE" > "$FAILOVER_SPARE.tmp" && mv "$FAILOVER_SPARE.tmp" "$FAILOVER_SPARE"
        sed -i "s|^$bad\$|$spare|" "$FAILOVER_ACTIVE"
        echo "$bad" >> "$FAILOVER_BAD"
        _failover_log "swapped: $bad -> $spare"
        swapped=$((swapped + 1))
    done
    _failover_log "swap summary: $swapped bad nodes replaced; $(wc -l < "$FAILOVER_SPARE") spares remaining"
    return 0
}

# ---------------------------------------------------------------------------
# failover_swap_one_blind
#
# Generic init-time crash with no specific bad hostname (e.g. the
# `set_determinism` `std::bad_alloc`). Just rotate ONE spare into the
# active set — pop the first active host, push to spare, pop spare,
# push to active.
# Returns 0 if success, 1 if no spares left.
# ---------------------------------------------------------------------------
failover_swap_one_blind() {
    local first_active
    first_active=$(head -n 1 "$FAILOVER_ACTIVE")
    [[ -n "$first_active" ]] || return 1
    failover_swap_in "$first_active"
}

# ---------------------------------------------------------------------------
# failover_run COMMAND...
#
# Run COMMAND, capturing its output to $FAILOVER_LOG_DIR/attempt-N.log.
# On non-zero exit, scrape the log for bad-node signatures and retry
# with swap-ins, up to $FAILOVER_MAX_RETRIES times.
# ---------------------------------------------------------------------------
failover_run() {
    local max=${FAILOVER_MAX_RETRIES:-3}
    local attempt=1
    local rc

    # If the command is `ezpz launch ...`, inject explicit topology args so
    # ezpz launch doesn't re-derive nhosts/ngpus from the original PBS aux
    # file (which still has 260/522 nodes — the spares we excluded). Without
    # this, _infer_topology computes ngpus=N_full*12 then trips
    # "ngpus must be > 0 and <= N_active*12, got N_full*12".
    #
    # IMPORTANT: ezpz launch's argparse uses UNDERSCORE forms for the long
    # flags (--nproc_per_node, --nnodes), not dash forms. Passing the dash
    # variant (--nproc-per-node) silently leaks the arg into cmd_to_launch
    # where mpiexec then rejects it with "unrecognized option". Use the
    # short flags (-n, -ppn) and the registered long forms to avoid that.
    local cmd=("$@")
    if [[ "${cmd[0]}" == "ezpz" && "${cmd[1]}" == "launch" ]]; then
        local ppn="${NGPU_PER_HOST:-12}"
        local nproc=$(( NHOSTS * ppn ))
        # FAILOVER_IDLE_TIMEOUT (default 1800s = 30min): if the launched
        # process emits no output for this long, ezpz launch sends SIGTERM
        # and exits 124. Catches silent collective hangs (e.g. the
        # 8479579 incident: 5h of W&B-heartbeat-alive-but-no-training-
        # metrics-output). Requires ezpz >= 0.15.1.
        local idle_timeout="${FAILOVER_IDLE_TIMEOUT:-1800}"
        cmd=(
            "${cmd[@]:0:2}"
            "--hostfile=$FAILOVER_ACTIVE"
            "--nnodes=$NHOSTS"
            "-ppn" "$ppn"
            "-n" "$nproc"
            "--timeout=$idle_timeout"
            "${cmd[@]:2}"
        )
    fi

    while (( attempt <= max + 1 )); do
        local logf="$FAILOVER_LOG_DIR/attempt-${attempt}.log"
        _failover_log "attempt ${attempt}/${max} — active=$(wc -l < "$FAILOVER_ACTIVE") nodes, spare=$(wc -l < "$FAILOVER_SPARE") nodes"
        _failover_log "logging to $logf"

        # Use stdbuf to keep tee'd output unbuffered, redirect both stderr and stdout.
        "${cmd[@]}" 2>&1 | tee "$logf"
        rc=${PIPESTATUS[0]}

        # ezpz launch's outer python wrapper sometimes exits 0 even when the
        # mpiexec child crashed (e.g. mpiexec --help dump, walltime SIGTERM).
        # When that happens we lose the bad-node signal. Always cross-check
        # the log for known crash patterns + the explicit "Execution finished
        # with N" trailer, and override rc accordingly.
        # Strip ANSI escape codes first — ezpz launch logs the trailer with
        # color codes baked in (`Execution finished with \x1b[1;36m127\x1b[0m`),
        # and a naive regex extracts '1' from the [1;36m prefix instead of
        # the real exit code (127). Pre-filter with sed.
        local inner_rc
        inner_rc=$(sed -r 's/\x1b\[[0-9;]*m//g' "$logf" 2>/dev/null | grep -oE "Execution finished with [0-9]+" | tail -1 | grep -oE "[0-9]+$")
        if [[ -n "$inner_rc" && "$inner_rc" != "0" ]]; then
            if (( rc == 0 )); then
                _failover_log "WARNING: shell exit 0 but log shows 'Execution finished with $inner_rc'; treating as failure"
                rc=$inner_rc
            fi
        elif (( rc == 0 )); then
            # Even without the trailer, mass-traceback / Connection-closed
            # patterns mean training crashed. Catch them.
            #
            # Threshold history:
            # - 8503077 (80B 2058N) only emitted 2 crash-pattern lines
            #   before SIGTERM cascaded, so the previous `> 5` threshold
            #   missed it. At very large rank counts, one bad node may
            #   only take a handful of ranks down before the wrapper
            #   tears everything down. Drop to `>= 1` — any of these
            #   patterns appearing means training crashed.
            # - Added EOFError (blendcorpus shuffle_idx mmap of a
            #   zero-length file) — fired in 8485515 80B 522N.
            local crash_lines
            crash_lines=$(grep -cE "RuntimeError: \[.*gloo.*\] Connection closed by peer|RuntimeError: \[.*gloo.*\] Timed out waiting|OutOfMemoryError|UR_RESULT_ERROR_OUT_OF_RESOURCES|died from signal|EOFError: No data left in file" "$logf" 2>/dev/null)
            if (( crash_lines >= 1 )); then
                _failover_log "WARNING: shell exit 0 but log has $crash_lines crash-pattern line(s); treating as failure (rc=1)"
                rc=1
            fi
        fi

        if (( rc == 0 )); then
            _failover_log "attempt ${attempt} succeeded (exit 0)"
            return 0
        fi

        # Walltime kills are NOT bad-node failures — exit -29 means walltime hit.
        # PBS reports exit -29 as bash exit 143 (128+15). Don't retry on those.
        # BUT: if we also found bad-node crash patterns, prefer the bad-node
        # path (don't bail on a true bad-node case just because the wallclock
        # signal also fired).
        # Walltime guard. Use the SAME crash-pattern set as the rc=0 case
        # above — otherwise a real bad-node failure that surfaced as
        # shell exit 143 (mpiexec SIGTERM'd after EOFError or
        # OutOfMemoryError) gets misclassified as a clean walltime kill
        # and the wrapper bails without retrying.
        if (( rc == 143 )); then
            local bad_crash_lines
            bad_crash_lines=$(grep -cE "RuntimeError: \[.*gloo.*\] Connection closed by peer|RuntimeError: \[.*gloo.*\] Timed out waiting|OutOfMemoryError|UR_RESULT_ERROR_OUT_OF_RESOURCES|died from signal|EOFError: No data left in file" "$logf" 2>/dev/null)
            if (( bad_crash_lines == 0 )); then
                _failover_log "attempt ${attempt} exited 143 (walltime / SIGTERM) — not a bad-node failure, no retry"
                return $rc
            fi
            _failover_log "attempt ${attempt} exited 143 but log has $bad_crash_lines bad-node lines — proceeding with retry"
        fi

        # exit 124 = ezpz launch idle-output watchdog (--timeout) fired.
        # The launched process emitted no output for FAILOVER_IDLE_TIMEOUT
        # seconds → ezpz sent SIGTERM. This catches silent hangs like
        # 8479579 (W&B alive but training metrics dead for 5h). Treat as
        # a bad-node failure: scrape, swap, retry. scrape_bad_nodes.py
        # likely won't find a specific hostname (the hang IS the silence)
        # so failover_swap_one_blind will rotate a spare.
        if (( rc == 124 )); then
            _failover_log "attempt ${attempt} exited 124 (ezpz idle-output watchdog tripped after ${FAILOVER_IDLE_TIMEOUT:-1800}s) — treating as silent-hang bad-node failure"
        fi

        _failover_log "attempt ${attempt} failed (exit $rc) — scraping for bad nodes"
        local bad_nodes
        bad_nodes=$(python3 "$(dirname "${BASH_SOURCE[0]}")/scrape_bad_nodes.py" "$logf" 2>/dev/null || true)

        if (( attempt > max )); then
            _failover_log "ERROR: max retries ($max) exhausted; giving up"
            return $rc
        fi

        if [[ -n "$bad_nodes" ]]; then
            _failover_log "bad nodes detected: $bad_nodes"
            failover_swap_in $bad_nodes || { _failover_log "swap failed; giving up"; return $rc; }
        else
            _failover_log "no specific bad node identified — rotating one spare in blindly"
            failover_swap_one_blind || { _failover_log "no spares left; giving up"; return $rc; }
        fi

        # Sanity check: still have nhosts_train active nodes
        local active_count
        active_count=$(wc -l < "$FAILOVER_ACTIVE")
        if (( active_count != NHOSTS )); then
            _failover_log "ERROR: active count drifted ($active_count != $NHOSTS); giving up"
            return $rc
        fi

        attempt=$((attempt + 1))
    done
}
