# Experiment Benchmark Reports

Benchmark results organized by experiment type and machine.

## Experiments

- [**agpt/**](agpt/) -- Dense AuroraGPT models (2B, 7B, 20B, 80B)
- [**moe/**](moe/) -- Mixture of Experts models (500M--10B)
- [**lr-finder/**](lr-finder/) -- Learning rate finder sweeps across models and optimizers

## Naming Convention

Report filenames follow: `YYYYMMDD-HHMMSS-<description>-n<nodes>.md`

| Field | Example | Meaning |
|-------|---------|---------|
| Date | `20260412` | Run date |
| Time | `002800` | Run start time (UTC-5) |
| Description | `smoke` | Test type (smoke, baseline, perf, prod-sim) |
| Nodes | `n2` | Number of compute nodes |
