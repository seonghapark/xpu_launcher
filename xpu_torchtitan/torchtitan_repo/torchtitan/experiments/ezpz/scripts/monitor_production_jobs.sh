#!/bin/bash
# monitor_production_jobs.sh -- emit one line per production-job STATE
# CHANGE, for use as a Monitor command (each stdout line -> one
# notification) or standalone in a terminal.
#
# Watches this user's agpt-{2b,20b,80b} PBS jobs and prints a line only
# when a job's state changes (Q->R, R->F/E, new job appears, job leaves
# the queue). Steady state is silent. On the first poll it prints the
# current roster once (so you have a baseline), then only deltas.
#
# Covers EVERY terminal transition, not just Q->R: a job vanishing from
# the queue (finished/killed) is reported too, so a crash is never
# silent.
#
# Usage:
#   # as a Monitor command (preferred -- notifications in chat):
#   bash torchtitan/experiments/ezpz/scripts/monitor_production_jobs.sh
#
#   # standalone in a terminal:
#   bash .../monitor_production_jobs.sh            # default 120s poll
#   POLL=60 USER=foremans bash .../monitor_production_jobs.sh
#   PATTERN='agpt-|eval-' bash .../monitor_production_jobs.sh   # widen match
#
# Env knobs:
#   USER    PBS user to watch         (default: $USER)
#   POLL    seconds between polls      (default: 120)
#   PATTERN egrep job-name filter      (default: agpt-(2b|20b|80b))
#   ONESHOT if set, print roster once and exit (no loop)

set -o pipefail

WATCH_USER="${USER:-$(whoami)}"
POLL="${POLL:-120}"
PATTERN="${PATTERN:-agpt-(2b|20b|80b)}"

# Snapshot: one "jobid:state:name" token per matching job, sorted.
# Resolves full Job_Name + job_state via `qstat -xf` per id (the `-u`
# summary truncates names; -xf is the authoritative source and also
# catches jobs that just moved to F/finished).
snapshot() {
    local ids id n s
    ids=$(qstat -u "$WATCH_USER" 2>/dev/null | tail -n +6 | awk '{sub(/\..*/,"",$1); print $1}')
    for id in $ids; do
        local rec; rec=$(qstat -xf "$id" 2>/dev/null) || continue
        n=$(echo "$rec" | grep -m1 "Job_Name"  | sed 's/.*= //')
        s=$(echo "$rec" | grep -m1 "job_state" | sed 's/.*= //')
        [[ "$n" =~ $PATTERN ]] || continue
        echo "${id}:${s}:${n}"
    done | sort
}

# Render a snapshot as a compact human line.
fmt() { echo "$1" | awk -F: '{printf "%s=%s ", $3, $2}'; }

prev=""
first=1
while true; do
    cur="$(snapshot | tr '\n' '|')"
    if [[ "$first" -eq 1 ]]; then
        echo "[$(date +%H:%M)] watching ${WATCH_USER} ${PATTERN} (poll ${POLL}s) -- roster: $(echo "$cur" | tr '|' ' ')"
        first=0
    elif [[ "$cur" != "$prev" ]]; then
        # Report per-job deltas so the line says WHAT changed.
        # Build assoc maps of jobid->state for prev and cur.
        declare -A P C
        P=(); C=()
        IFS='|' read -ra parr <<< "$prev"; for e in "${parr[@]}"; do [[ -n "$e" ]] || continue; jid=${e%%:*}; st=$(echo "$e"|cut -d: -f2); P[$jid]=$st; done
        IFS='|' read -ra carr <<< "$cur";  for e in "${carr[@]}"; do [[ -n "$e" ]] || continue; jid=${e%%:*}; st=$(echo "$e"|cut -d: -f2); nm=$(echo "$e"|cut -d: -f3); C[$jid]=$st; CN[$jid]=$nm; done
        msg=""
        for jid in "${!C[@]}"; do
            if [[ -z "${P[$jid]:-}" ]]; then msg+="${CN[$jid]}(${jid}) NEW->${C[$jid]}; "
            elif [[ "${P[$jid]}" != "${C[$jid]}" ]]; then msg+="${CN[$jid]}(${jid}) ${P[$jid]}->${C[$jid]}; "; fi
        done
        for jid in "${!P[@]}"; do
            [[ -z "${C[$jid]:-}" ]] && msg+="job ${jid} LEFT queue (was ${P[$jid]}); "
        done
        echo "[$(date +%H:%M)] ${msg:-state changed}"
        unset P C CN
    fi
    prev="$cur"
    [[ -n "${ONESHOT:-}" ]] && break
    sleep "$POLL"
done
