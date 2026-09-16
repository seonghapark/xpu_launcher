# Production Queue-Wait Analysis (Aurora `small`)

> Snapshot: 2026-06-24. Queue wait — not training throughput — is the
> dominant cost on the 512N canonical chains right now. This page
> records the data + the scheduler diagnosis so we stop re-deriving it.

## TL;DR

- **512N is queue-starved.** Both 512N canonical chains (2B + 20B) have
  sat in `small` for **~20 days** (since 2026-06-04) without a slot.
- **256N cycles in hours-to-days.** 256N (260-node) jobs land far more
  often; the worst 256N wait we recorded was ~9 days, most are <1 day.
- **Root cause is pure contention, confirmed by PBS** — not a hold, not
  a bad dependency, not an unsatisfiable request. The 522-node ask
  rarely fits in `small` alongside everyone else.
- **Do NOT re-submit the stuck 512N jobs.** They've accrued ~18 days of
  scheduling priority (`eligible_time`); a fresh submit resets that to
  zero and goes to the back of the queue. Re-submitting makes it worse.

## UPDATE 2026-06-26 -- `at_queue`, and even 256N is now blocked

The "pure contention" framing above is incomplete. On 2026-06-26 the
blocking PBS comment was `Not Running: Insufficient amount of resource:
at_queue` -- a queue-level resource limit, not simply "not enough free
nodes". **Evidenced** observations (not theory):

- **`small` is live and DID start 256N jobs** -- seven 256N/288N jobs
  from OTHER projects (`alcf_training` x5, `AI4SRM`, a physics project)
  started 09:42-11:16 on 2026-06-26.
- **AuroraGPT had 0 jobs running in `small`** (1 running machine-wide)
  the entire time, despite ~3000 physically-free nodes.
- So the freed `small` capacity went to other projects' jobs ahead of
  AuroraGPT's. AuroraGPT is **losing the scheduling race for `small`**,
  not waiting on raw node availability.
- **It is NOT about eligibility/age**: a fresh throwaway 256N submit
  (eligible_time ~0) shows the same `at_queue` as the canonical
  cont (eligible_time ~56h). So kill+resubmit of a starved canonical job
  is pointless -- it burns ~56h of accrued priority and hits the same
  wall. (Verified by submitting `agpt-2b-n256-FRESHTEST` alongside.)

**What is NOT the cause** (checked + ruled out 2026-06-26): a hold
(`Hold_Types = n`), a bad dependency, an unsatisfiable request, the
Jun-29 PM reservation (starts in ~57h; too far out to be draining nodes
now), the project allocation (burn_ratio 0.29 -- only 29% of INCITE-2026
used, healthy), or a job-config error.

**What is NOT yet known** (needs scheduler-policy / ALCF visibility, not
derivable from `qstat`): the exact mechanism by which other projects
outrank an INCITE allocation for `small` right now. Candidates
(UNCONFIRMED -- do not state as fact): a higher-priority training/
reservation queue for `alcf_training`, a recent-usage decay term, or an
AuroraGPT priority standing worth raising with ALCF. An earlier draft of
this note guessed "WFP fair-share"; that was speculation from seeing
`enable_wfp=1` on the job and should not be trusted without confirmation.

**Implication for tactics:** more submits (sneaks, re-queues) cannot beat
this -- they share the project's scheduling standing. The levers are
time (the race shifts), the pre-PM drain (forces placement of backlog),
or a PI->ALCF priority question. The multi-chain umbrella in `medium`
(uncapped, 0 assigned) is a parallel bet but waits on the same project
standing.

## Live data (from PBS `qtime`, 2026-06-24)

| Job | Trajectory | Nodes | Queued since | Wait so far | PBS comment |
|-----|------------|------:|--------------|------------:|-------------|
| 8521631 | 2B 512N cont10 | 522 | 2026-06-04 11:03 | **~20 d** | Not enough free nodes available |
| 8521632 | 20B 512N cont11 | 522 | 2026-06-04 11:24 | **~20 d** | Not enough free nodes available |
| 8558531 | 2B 256N cont12 | 260 | 2026-06-24 07:19 | <1 d | (cycling) |
| 8558548 | 20B 256N | 260 | 2026-06-24 07:45 | <1 d | (cycling) |

`eligible_time` at snapshot: 8521631 = 430 h, 8521632 = 375 h — i.e.
these jobs have been accruing priority for ~16-18 days. That priority
is the asset that will eventually win them a slot; it is destroyed by
re-submission.

The `H`-held continuations (8534294/95, 8558532, 8558549) are NOT
waiting on the queue — they are `afterany`-held behind their
predecessor and only become eligible once it finishes.

## Why 512N specifically

`qstat -Qf small` at snapshot: `resources_assigned.nodect = 4792`
(nodes already committed to other running jobs). A 522-node *scatter*
placement needs 522 free hosts simultaneously; with ~4.8k nodes
committed elsewhere that window is rare. 256N (260 nodes) fits into
gaps far more often, which is exactly why the 256N chains keep
advancing while 512N stalls.

512N *is* schedulable — earlier 512N dispatches did run (e.g. 8505258,
8509393 on the 20B chain) — it's just infrequent.

## Is it worth re-submitting fresh 512N jobs? — NO

Checked explicitly because it's the tempting move. PBS shows:

- `Hold_Types = n` (no hold), `depend = beforeany:<cont>` only (the
  job is the *predecessor*, not blocked by anything).
- `comment = Not Running: Not enough free nodes available`.

So nothing is wrong with the jobs; they are simply waiting for 522
free nodes. A re-submit:

1. Starts at `eligible_time = 0`, behind the current jobs that have
   ~18 days accrued.
2. Breaks the `afterany` continuation chain (the held conts reference
   the current jobids).

Re-submitting would *lengthen* the wait, not shorten it.

## Real levers (if 512N progress becomes urgent)

1. **Project reservation** — a `pbs_rstat`/reservation for a 512N+
   block is the only reliable way to guarantee the nodes. Worth raising
   at the AuroraGPT sync if 512N throughput matters near-term.
2. **Consolidate on 256N** — 256N gets slots; per-token it is the
   better comparator anyway (the 256N chains carry the bulk of the
   token progress). The 512N chains exist for the canonical large-batch
   trajectory, but if wall-clock progress is the priority, 256N
   delivers it.
3. **Smaller 512N walltime** — shorter jobs backfill into gaps more
   easily, at the cost of more frequent restarts (and more
   `set_determinism` init-crash exposure on 20B).

## Ongoing data collection

`failover_lib.sh::failover_log_queue_wait` (added 2026-06-24) appends
every dispatch's actual queue wait to `logs/queue_wait.csv` at job
start (qtime -> stime), so we have a durable record after jobs age out
of `qstat`. Columns:
`jobid,jobname,nodes,queue,qtime,stime,wait_seconds,wait_hours`.
Aggregate it over time to track whether the 512N starvation is getting
better or worse.
