# Pre-#3623 checkpoints can't resume on current code: optimizer state-dict format migration

> **Root cause, evidence-based (2026-06-28).** Production checkpoints saved
> before upstream PR **#3623** ("[Checkpointer] Remove the dependencies on
> PyTorch distributed state_dict APIs", commit `772dd1b6c`) cannot be resumed
> by current `ezpz` HEAD. The crash is
> `RuntimeError: Missing key in checkpoint state_dict:
> optimizer.param_groups.<param>.weight.fused`. This is an **optimizer
> state-dict FORMAT migration**, not a key rename and NOT FusedQKVLinear.

## What actually changed (verified, not inferred)

`git diff 1263e5a1 HEAD -- torchtitan/components/optimizer.py` shows the
optimizer state-dict API was swapped:

```
- from torch.distributed.checkpoint.state_dict import (
-     get_optimizer_state_dict, set_optimizer_state_dict )
+ from torchtitan.components.checkpoint_utils import (
+     get_flat_optim_state_dict, load_flat_optim_state_dict )
```

(`optimizer.py:301` `get_flat_optim_state_dict(optim)`, `:309`
`load_flat_optim_state_dict(optim, state_dict)`.) Two upstream commits between
`1263e5a1` (pre, works) and HEAD (post, crashes):

- **`772dd1b6c` #3623** — replaces PyTorch's `get/set_optimizer_state_dict`
  with torchtitan's own `get_flat_optim_state_dict` / `load_flat_optim_state_dict`
  (new `checkpoint_utils.py`, +239 lines). This is the format change.
- **`632f67f12` #3269** — mixed-optimizer support; `param_groups` now carry a
  `param_names` list of canonical FQNs.

The new flat representation keys per-parameter optimizer state differently
(the `.fused` suffix the crash names is part of the new flat layout, applied
to ALL params incl. `tok_embeddings`). Old checkpoints were written by the
nested `get_optimizer_state_dict` format and lack those keys.

## What it is NOT (theories ruled out, with evidence)

- **NOT FusedQKVLinear** (#3714 / `70dd94551`). Those model-state hooks
  (`_merge_qkv_on_load` at attention.py:832) only touch MODEL state (wq/wk/wv
  <-> wqkv) and would never put `.fused` on `tok_embeddings`. Red herring.
- **NOT the optimizer `implementation="fused"` flag.** `optimizer.py`
  `implementation` defaulted to `"fused"` in BOTH `1263e5a1` and HEAD --
  unchanged, so it's not the era difference.

## Status / why it's not blocking

Production clones (`agpt-{2b-v2,20b-v2,20b-n256}`) are **pinned pre-#3623 at
`1263e5a1`** + this session's ckpt-safe infra (see
[[feedback_clone_pull_can_break_ckpt_resume]]). They resume the existing
checkpoints cleanly -- proven 2026-06-28: the 2B-256N chain advanced
86,200 -> 86,674 (sneak 8572612) and the load-test 8570407 loaded
step-86,200 in 32s. **No production chain is blocked.** The only cost of
staying pinned is missing ~4 weeks of upstream (incl. the 80B optimizer
findings, RoPE refactor, etc.) on the production training code.

## Options to eventually return clones to current code (NOT yet done)

1. **Stay pinned (current).** Zero risk; clones keep resuming. Revisit only
   if a current-HEAD feature is needed in production training. Cheapest.
2. **Optimizer-state migration shim (HARD, risky).** Convert the old nested
   DCP optimizer state to the new flat layout on load. Reconstructing the flat
   representation from old shards is non-trivial, and a subtle error loads
   wrong Adam momentum/variance -> silently corrupts training (no crash).
   Would need bit-identical loss+grad_norm validation pre/post on a real
   checkpoint before trusting. High effort, high blast radius.
3. **Fresh-start the chains on current code.** Only if the science wants the
   upstream model/optimizer changes badly enough to eat the lost tokens.
   Not warranted for chains at ~93% of target.

**Recommendation:** option 1 (stay pinned) until there's a concrete need for
a current-HEAD feature in production. If/when option 2 is attempted, do it as
its own focused effort with a debug-scaling A/B that asserts identical
loss/grad_norm resuming a real ckpt the old way vs the migrated way -- do not
ship it on inference about the flat format.

## Reproduce the diagnosis

```bash
git diff 1263e5a1 HEAD -- torchtitan/components/optimizer.py | grep -E 'state_dict import|flat_optim'
git log --oneline 1263e5a1..HEAD -- torchtitan/components/optimizer.py | grep -iE 'flat|#3623|#3269'
```
