# Summaries

Periodic retrospectives covering the project at a higher level than
[`journal.md`](../journal.md) (which is a per-session running log).
Each summary names the explicit date range it covers in its filename
(`YYYY-MM-DD_to_YYYY-MM-DD.md`) and in its title.

## Quarterly / program reports

Higher-level rollups for external (INCITE / program) reporting,
synthesized from the two-week summaries below.

| Period | Report | Headline |
|---|---|---|
| 2026 Q2 (Apr 1 -> Jun 30) | [INCITE Quarterly Report](2026-Q2-incite.md) | 2B base pre-training COMPLETE (4.674T tokens); 20B leading per token; 80B launched; RL (SFT+GRPO) end-to-end on XPU; ~29% Year-2 Aurora burn |

## Index

| Period | Summary | Headline |
|---|---|---|
| 2026-04-12 → 2026-04-27 | [Two-Week Summary](2026-04-12_to_2026-04-27.md) | 291 commits — built LR finder + scaling study + production training + optimizer competition platform |
| 2026-05-08 → 2026-05-22 | [Two-Week Summary](2026-05-08_to_2026-05-22.md) | 51 commits — filed first upstream PyTorch PR (xccl `_set_pg_timeout` dispatch), 4 upstream syncs, 80B bad-node failover wrapper, silent-hang detection fix |
| 2026-05-22 → 2026-05-29 | [One-Week Summary](2026-05-22_to_2026-05-29.md) | 80 commits — 2 upstream PRs (1 closed, 1 superseded by draft #3450), 5 upstream syncs absorbed, 20B 512N HSn 0.579→0.635, failover wrapper hardening, eval/chart infrastructure overhaul |
| 2026-06-05 → 2026-06-12 | [One-Week Summary](2026-06-05_to_2026-06-12.md) | ~140 commits — SFT 2B × tulu_math_uc_mix completed (729 steps), GRPO 8N production run done, 4 upstream syncs (50-53), PR #14 review with full A/B + 1.94× speedup confirmed, vLLM-XPU sibling venv, olmo-mix-1124 8N path documented, ambivalent + Iosevka chart restyle, os._exit hang fix |
| 2026-06-12 to 2026-06-26 | [Two-Week Summary](2026-06-12_to_2026-06-26.md) | 151 commits -- 80B grad-path NaN root-caused (2 triggers: LBS>1 + dp_degree>186), TP=4/LBS=1/bf16 stable corner confirmed + stress-tested to 4-16x batch (512N/1024N/2048N global-batch sims), GRPO RL end-to-end on XPU via TRL vllm-serve, native ezpz launch --auto-retry scripts, 4 upstream syncs (56-59), PR #14 merged, blendcorpus index-race + barrier lesson |
