# Production Training — agpt 2B @ 1024 nodes

> **Status: crashed at startup (2026-05-04).** First-ever 1024N attempt
> (8463182) ran for 221s then died with `MemoryError: std::bad_alloc`
> inside `torch.distributed.broadcast` during the `set_determinism`
> distributed-init phase. Likely an init-time OOM at 12,288 ranks
> (matches the 20B 1024N crash from the same queue release). Needs a
> smaller-scale repro (768N? 896N?) before resubmitting.

## v2 — 2B @ 1024N — SophiaG LR=2.28e-5 (fp32 master)

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/` |
| Submit script | `scripts/submit_agpt_2b_aurora_venv.sh` (deprecated 2026-06-24; this dead n1024 attempt predates the failover wrapper) |
| Stack | torch 2.13 venv (yeet-env tarball mode) |
| Compile | on |
| GBS | 24,576 (LBS=2 × 1024 nodes × 12 GPUs) |
| Total tokens | 4.67T target |
| Checkpoint dir | `outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n1024-gbs24576` (will create on first save) |

### Progress

| Job ID | Date | Walltime | Steps | Status |
|--------|------|---------:|------:|--------|
| [`8463182`](#log-8463182) | 2026-05-04 | 12h | — | **Crashed at startup** (221s, exit 143). `MemoryError: std::bad_alloc` in `torch.distributed.broadcast` during `set_determinism`. |

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8463182"></a>`8463182` | `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/agpt-2b-n1024-v2.o8463182` |

This is an *independent* trajectory from the canonical 512N chain
([n512/](../n512/README.md)) — it writes to a different ckpt dir
(`gbs24576` vs `gbs12288`) and would start fresh from step 0. Useful
as a scaling experiment, not as a chain extension.

