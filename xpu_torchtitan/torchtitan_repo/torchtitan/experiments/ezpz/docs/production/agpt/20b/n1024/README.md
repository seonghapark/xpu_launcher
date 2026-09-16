# Production Training — agpt 20B @ 1024 nodes

> **Status: crashed at startup (2026-05-04).** First-ever 1024N attempt
> (8463183) ran for 211s then died with `signal 11` (SIGSEGV) during
> the `set_determinism` distributed-init phase. Likely an init-time
> OOM at 12,288 ranks (matches the 2B 1024N crash from the same
> queue release). Needs a smaller-scale repro (768N? 896N?) before
> resubmitting.

## v2 — 20B @ 1024N — SophiaG LR=2.28e-5 (fp32 master)

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/` |
| Submit script | `scripts/submit_agpt_20b_aurora_venv.sh` (deprecated 2026-06-24; this dead n1024 attempt predates the failover wrapper) |
| Stack | torch 2.13 venv (yeet-env tarball mode) |
| Compile | on |
| GBS | 24,576 (LBS=2 × 1024 nodes × 12 GPUs) |
| Total tokens | 4.67T target |
| Checkpoint dir | `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n1024-gbs24576` (will create on first save) |

### Progress

| Job ID | Date | Walltime | Steps | Status |
|--------|------|---------:|------:|--------|
| [`8463183`](#log-8463183) | 2026-05-04 | 12h | — | **Crashed at startup** (211s, exit 143). `rank 4732 died from signal 11` during `set_determinism` distributed-init. |

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8463183"></a>`8463183` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n1024-v2.o8463183` |

This is an *independent* trajectory from the canonical 512N chain
([n512/](../n512/README.md)) — it writes to a different ckpt dir
(`gbs24576` vs `gbs12288`) and would start fresh from step 0. Useful
as a scaling experiment, not as a chain extension.

