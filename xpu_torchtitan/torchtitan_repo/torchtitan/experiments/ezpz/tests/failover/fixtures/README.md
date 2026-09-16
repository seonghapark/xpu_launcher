# Failover wrapper test fixtures

Each fixture is a synthetic `attempt-N.log` snippet that reproduces one
of the failure modes we've seen in production, with the same structure
the wrapper sees (ANSI codes included). Used by `tests/failover/run_tests.sh`
to verify the `failover_run` exit-code detection logic without needing
to dispatch a real PBS job.

Each `.log` file ends with the exact byte sequence that PBS writes when
`ezpz launch` returns: a `[2026-...] Execution finished with [1;36mN[0m.`
trailer (or no trailer if SIGTERM came from outside ezpz launch's
control).

| Fixture | Failure mode | Expected wrapper decision |
|---------|--------------|---------------------------|
| `clean_success.log` | Training completed normally, exit 0 | succeeded |
| `gloo_cascade_mass.log` | Bad node, 3074 `Connection closed by peer` lines | rc=1 (mass crash detect) → retry |
| `gloo_cascade_few.log` | Bad node, only 2 crash lines (small cluster cascade) | rc=1 (>= 1 threshold) → retry |
| `silent_hang.log` | No output for 30+ min, watchdog fires | rc=124 → retry (silent hang) |
| `walltime_clean.log` | exit 143, no crash patterns | rc=143 no-retry (walltime guard) |
| `walltime_with_crash.log` | exit 143 + crash pattern in log | rc=143 → retry (bad-node override) |
| `mpiexec_help_dump.log` | mpiexec dumped --help (flag rejection) | rc=1 → retry |
| `set_determinism_oom.log` | `MemoryError: std::bad_alloc` in init | rc=1 → retry |
| `blendcorpus_eoferror.log` | EOFError from empty shuffle_idx cache | rc=1 → retry |

The fixture content is BYTE-IDENTICAL to what production logs contain
(ANSI escape \x1b included as literal bytes). To inspect: `cat -A`.
