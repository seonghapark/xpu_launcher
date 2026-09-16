# agpt_2b Loss Competition

See [docs/competitions/](../docs/competitions/) for the full
leaderboard, experiment tracking, and modifications log.

**W&B Report:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

## Quick Start

```bash
# Run a single config
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt --config speedrun_2b_muon

# Submit to PBS (2 nodes, 3h)
qsub -l select=2 -N speedrun_2b_muon -v CONFIG=speedrun_2b_muon \
    torchtitan/experiments/ezpz/competition/submit_run.sh
```
